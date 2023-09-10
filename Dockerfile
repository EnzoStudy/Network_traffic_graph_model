
FROM pytorch/pytorch

   

# app 디렉토리 생성

RUN mkdir -p /home
RUN mkdir -p /home/dev

#Docker 이미지 내부에서 RUN, CMD, ENTRYPOINT의 명령이 실행될 디렉터리를 설정합니다.
WORKDIR /home/dev

# # 현재 디렉터리에 있는 파일들을 이미지 내부 /app 디렉터리에 추가함
ADD  .  /home/dev

RUN apt-get update -y

### Github Source 파일 다운로드    
RUN apt install git -y
#RUN git clone https://github.com/EnzoStudy/Network_traffic_graph_model.git


## pip install 
RUN cd /home/dev
RUN pip install -r requirements.txt
RUN pip install pandas 
RUN pip install matplotlib 
RUN pip install torch_geometric 
RUN pip install torch_scatter -f https://data.pyg.org/whl/torch-2.0.1%2Bcu118.html 
RUN pip install seaborn 
RUN pip install tensorflow
RUN pip install keras
RUN pip install tensorboardX 