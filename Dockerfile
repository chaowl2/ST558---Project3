FROM rocker/r-ver:4.3.1

# System libraries needed by R packages
RUN apt-get update && apt-get install -y \
    libssl-dev libcurl4-openssl-dev libxml2-dev \
    libfontconfig1-dev libfreetype6-dev libpng-dev \
    libtiff5-dev libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages used in api.R
# Install R packages used in api.R
RUN R -e "install.packages(c('tidyverse','tidymodels','plumber','janitor','skimr','ranger','ggplot2','rsample','doParallel','scales','patchwork'), repos = 'https://cloud.r-project.org')"




# Create working directory in container
WORKDIR /app

# Copy API and data files into the container
COPY api.R /app/api.R
COPY data/ /app/data/

# Expose plumber API port
EXPOSE 8000

# Start the Plumber API when the container runs
CMD ["R", "-e", "pr <- plumber::plumb('api.R'); pr$run(host='0.0.0.0', port=8000)"]
