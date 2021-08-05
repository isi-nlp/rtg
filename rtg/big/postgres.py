#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/12/20

import shutil
import socket
import subprocess
from pathlib import Path

from rtg import log


def download_file(url, path, fair_on_error=True):
    try:
        log.info(f"Trying to download {url} --> {path}")
        import requests
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)
        return True
    except:
        if fair_on_error:
            raise
        else:
            log.exception(f"Could NOT download {url} --> {path}")


def setup_jdbc():
    # add postgres jdbc driver
    import pyspark
    PG_JDBC_URL = 'https://jdbc.postgresql.org/download/postgresql-42.2.14.jar'
    jars_dir = Path(pyspark.__file__).parent / 'jars'
    if jars_dir.exists():
        pg_jdbc_jars = list(jars_dir.glob("postgresql-*.jar"))
        if pg_jdbc_jars:
            log.info(f'Located JDBC jar for postgres: {pg_jdbc_jars}')
        else:
            jar_path = jars_dir / (PG_JDBC_URL.split('/')[-1])
            download_file(PG_JDBC_URL, jar_path, fair_on_error=False)
    else:
        log.warning("pyspark jars are not detected. "
                    "You may need to manually configure postgres JDBC to spark config")


def run_command(cmd_line: str, fail_on_error=True):
    log.info(f'RUN:: {cmd_line}')
    proc = subprocess.run(cmd_line, shell=True, capture_output=True, text=True)
    if proc.returncode == 0:
        log.info(f"STDOUT={proc.stdout}")
        if proc.stderr:
            log.warning(f"STDERR={proc.stderr}")
    else:
        msg = f'CMD={cmd_line}\nCODE={proc.returncode}\nSTDOUT={proc.stdout}\nSTDERR={proc.stderr}'
        if fail_on_error:
            raise Exception('subprocess failed\n' + msg)
        else:
            log.warning(msg)
    return proc.returncode == 0


class PostgresServer:

    def __init__(self, db_dir: Path, log_dir: Path, dbname='rtg', superuser='postgres', port=5433):
        self.db_dir = db_dir
        self.init_flag = db_dir / "_RTG_INIT_SUCCESS"
        self.db_name = dbname
        self.port = port
        self.url = f"jdbc:postgresql://localhost:{self.port}/{self.db_name}"
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        log_file = log_dir / f'pgdb.{socket.gethostname()}.{port}.log'
        required_cmds = ['initdb', 'pg_ctl', 'createuser', 'createdb']
        # -o "-F -p 5433"    to change port
        for cmd in required_cmds:
            if not shutil.which(cmd):
                raise Exception(f"{cmd} is required but it is not found.")

        self.setup_seq = [
            f'initdb -D {db_dir} ',  # initialize
            f'pg_ctl start -o "-F -p {port}" -D {db_dir}  -l {log_file} ',  # start
            f'createuser -p {port} -s {superuser}',  # at least one superuser is needed
            f'createdb -p {port} {dbname}'  # create db
        ]
        self.start_cmd = f'pg_ctl start -o "-F -p {port}" -D {db_dir} -l {log_file}'
        self.stop_cmd = f'pg_ctl stop -o "-F -p {port}" -D {db_dir}'

    def write_df(self, df, table_name: str, mode="overwrite"):

        log.info(f"writing dataframe to {self.url}; table={table_name} mode={mode}")
        return (df.write
                .mode(mode)
                .format("jdbc")
                .option("url", self.url)
                .option("dbtable", table_name)
                .option("driver", "org.postgresql.Driver")
                .save())

    def read_df(self, spark, table_name: str):
        return (spark.read
                .format("jdbc")
                .option("url", self.url)
                .option("dbtable", table_name)
                .option("driver", "org.postgresql.Driver")
                .load())

    def start(self):
        if self.init_flag.exists():
            log.info("Going to start db")
            run_command(self.start_cmd, fail_on_error=True)
        else:
            log.info("Going to set up db")
            for cmd in self.setup_seq:
                run_command(cmd_line=cmd, fail_on_error=True)
            self.init_flag.touch()

    def stop(self):
        log.info("Stopping db")
        if not run_command(self.stop_cmd, fail_on_error=False):
            log.warning("Could not stop database. You may need to stop it manually")


def main():
    server = PostgresServer(db_dir=Path("tmpdb/dbdir"), log_dir=Path('tmpdb/logs'))
    server.start()

    # server.stop()


if __name__ == '__main__':
    main()
