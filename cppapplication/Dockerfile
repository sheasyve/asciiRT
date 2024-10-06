# Use the GCC base image
FROM gcc:latest

# Install dependencies (cmake and Eigen)
RUN apt-get update && apt-get install -y cmake libeigen3-dev

# Set up the working directory
WORKDIR /cppapplication

# Copy all source code into the Docker image
COPY . .

# Create and move into the build directory
WORKDIR /cppapplication/build

# Configure the project using CMake
RUN cmake ..

# Clean and build the project
RUN make clean
RUN make

# List the contents of the build directory (for debugging purposes)
RUN ls -l /cppapplication/build

# Specify the entry point (run the executable)
CMD ["./ascii_rt"]
