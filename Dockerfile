FROM python:3.7-slim

# Install C dependencies.
# It's important to do apt-get update and install in the
# same command.  It's more efficient to only do it once.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gcc \
      libmariadb-dev \
      libmariadb-dev-compat

# Update pip
RUN python -m pip install --upgrade pip

# Create the application directory and point there
# (WORKDIR will implicitly create it)
WORKDIR /app/

# Install all of the Python dependencies.  These are
# listed, one to a line, in the requirements.txt file,
# possibly with version constraints.  Having this as
# a separate block allows Docker to not repeat it if
# only your application code changes.
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Copy in the rest of the application.
COPY . .

# Specify what port your application uses, and the
# default command to use when launching the container.
EXPOSE 8000
CMD /usr/local/bin/gunicorn main:app -w 2 -b :8000
