FROM python:3.11-slim
WORKDIR /srv
RUN pip install --no-cache-dir tensorflow-cpu flask pillow numpy
COPY main.py .
COPY champion.keras .
RUN useradd -m app && chown -R app /srv
USER app
EXPOSE 5000
CMD ["python", "main.py"]
