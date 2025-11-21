FROM gurobi/python

#install julia and dependencies
RUN apt update && apt -y install  wget
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.1-linux-x86_64.tar.gz
RUN tar zxvf julia-1.8.1-linux-x86_64.tar.gz
COPY install.jl install.jl
RUN julia-1.8.1/bin/julia install.jl


COPY . /


COPY run.sh run.sh

CMD ["julia-1.8.1/bin/julia", "-i", "server.jl"]

EXPOSE 8888