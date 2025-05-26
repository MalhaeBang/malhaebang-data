*만약 docker에서 data.csv가 directory로 인식해서 꼬인다면*
$ docker compose down --volumes --rmi all
$ docker system prune -af
$ docker compose up --build
