FROM gcc:latest
RUN apt-get update && apt-get install -y cmake libeigen3-dev
WORKDIR /cppapplication
COPY . .
WORKDIR /cppapplication/build
RUN cmake ..
RUN make clean
RUN make
RUN ls -l /cppapplication/build
CMD ["./ascii_rt"] 
