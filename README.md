
#### 👀 만약 docker가 data.csv를 directory로 인식해버린다면

$ docker compose down --volumes --rmi all

$ docker system prune -af

$ docker compose up --build
