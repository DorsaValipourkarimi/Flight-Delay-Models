import kagglehub

# Download latest version
path = kagglehub.dataset_download("nalisha/flight-delay-and-cancellation-data-1-million-2024")

print("Path to dataset files:", path)