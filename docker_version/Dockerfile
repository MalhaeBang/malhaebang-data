FROM python:3.10

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    unzip \
    fonts-nanum \
    build-essential \
    libnss3 \
    libxss1 \
    libappindicator3-1 \
    libasound2 \
    libatk1.0-0 \
    libcups2 \
    libx11-xcb1 \
    libxcomposite1 \
    libxrandr2 \
    libxdamage1 \
    libgbm1 \
    libgtk-3-0 \
    libxext6 \
    libxfixes3 \
    libfontconfig1 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libglib2.0-0 \
    libgconf-2-4 \
    libdbus-1-3 || apt-get install -yf

# Chrome 설치
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get update --fix-missing && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb || apt-get install -yf && \
    rm google-chrome-stable_current_amd64.deb

# chromedriver 복사 및 권한 설정
COPY chromedriver-linux64/chromedriver /usr/local/bin/chromedriver
RUN chmod +x /usr/local/bin/chromedriver

# Java 설치
RUN apt-get update && apt-get install -y default-jdk
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH=$JAVA_HOME/lib/server
RUN ln -sf $JAVA_HOME/lib/server/libjvm.so /usr/lib/libjvm.so
ENV PATH="$JAVA_HOME/bin:$PATH"

# 작업 디렉토리 설정 및 전체 코드 복사
WORKDIR /app
# COPY . /app
COPY requirements.txt .
COPY *.py .
COPY templates/ ./templates/
COPY data.csv .

# 파이썬 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 포트 오픈
EXPOSE 8002

# 실행 (main 실패해도 app 실행되도록 분리)
CMD python main.py || true && python app.py
