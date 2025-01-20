import csv
import random

# Parameters
output_file = "vectors_input.csv"
num_vectors = 100  # Number of vectors to generate
num_dimensions = 5  # Number of dimensions per vector

# Generate fake vector data
def generate_vectors(num_vectors, num_dimensions):
    data = []
    for i in range(num_vectors):
        vector = [i]  # Start with an ID
        vector.extend([round(random.uniform(0.0, 1.0), 2) for _ in range(num_dimensions)])
        data.append(vector)
    return data

# Write data to CSV
def write_csv(file_name, data, num_dimensions):
    header = ["id"] + [f"dim{d+1}" for d in range(num_dimensions)]
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

if __name__ == "__main__":
    vectors = generate_vectors(num_vectors, num_dimensions)
    write_csv(output_file, vectors, num_dimensions)
    print(f"Fake vector data written to {output_file}")
