#include <mpi.h>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "lodepng.h"

using std::string;

typedef std::vector<double> Vector;

enum MessageType { kUpperOverlap, kLowerOverlap, kLowerHalf };

Vector LoadImage(const string& filename, unsigned& width, unsigned& height);
void SaveImage(Vector& image, unsigned width, unsigned height,
               const string& filename, int id);
void compute(Vector& image, unsigned width, unsigned height, int id);

int main() {
  unsigned width, height;

  Vector image = LoadImage("image.png", width, height);

  std::cout << "The dimension of the image is " << height << "x" << width
            << ".\n";

  int id, num_procs;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  compute(image, width, height, id);

  SaveImage(image, width, height, "result.png", id);

  MPI_Finalize();

  return 0;
}

void compute(Vector& image, unsigned width, unsigned height, int id) {
  const unsigned kNumIndepIters = 10;
  assert(height / 2 > kNumIndepIters);
  if (id > 1) return;

  Vector image_new(width * height);

  for (unsigned sweep = 0; sweep < height; ++sweep) {
    unsigned start = id == 0 ? 1 : height / 2 - kNumIndepIters + 1;
    unsigned end = id == 0 ? height / 2 + kNumIndepIters : height - 1;
    for (unsigned iteration = 0; iteration < kNumIndepIters; ++iteration) {
      for (unsigned y = start; y < end; ++y) {
        for (unsigned x = 1; x < width - 1; ++x) {
          image_new[y * width + x] =
              (image[(y - 1) * width + x] + image[(y + 1) * width + x] +
               image[y * width + x + 1] + image[y * width + x - 1] +
               image[y * width + x]) /
              5;
        }
      }

      if (id == 0) {
        --end;
      } else {
        ++start;
      }
      swap(image_new, image);
    }
    if (id == 0) {
      MPI_Send(&image[height / 2 * width - kNumIndepIters * width],
               kNumIndepIters * width, MPI_DOUBLE, 1, kUpperOverlap,
               MPI_COMM_WORLD);
      MPI_Recv(&image[height / 2 * width], kNumIndepIters * width, MPI_DOUBLE,
               1, kLowerOverlap, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      MPI_Recv(&image[height / 2 * width - kNumIndepIters * width],
               kNumIndepIters * width, MPI_DOUBLE, 0, kUpperOverlap,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&image[height / 2 * width], kNumIndepIters * width, MPI_DOUBLE,
               0, kLowerOverlap, MPI_COMM_WORLD);
    }
  }
}

void SaveImage(Vector& image, unsigned width, unsigned height,
               const string& filename, int id) {
  switch (id) {
    case 0:
      MPI_Recv(&image[height / 2 * width], height / 2 * width, MPI_DOUBLE, 1,
               kLowerHalf, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      break;
    case 1:
      MPI_Send(&image[height / 2 * width], height / 2 * width, MPI_DOUBLE, 0,
               kLowerHalf, MPI_COMM_WORLD);
      // fall through
    default:
      return;
  }

  std::vector<unsigned char> pixels(width * height * 4);
  for (unsigned i = 0; i < width * height; ++i) {
    pixels[4 * i] = pixels[4 * i + 1] = pixels[4 * i + 2] =
        (unsigned char)image[i];
    pixels[4 * i + 3] = 255;
  }

  unsigned error = lodepng::encode(filename, pixels, width, height);

  // if there's an error, display it
  if (error)
    std::cout << "encoder error " << error << ": " << lodepng_error_text(error)
              << std::endl;
}

Vector LoadImage(const string& filename, unsigned& width, unsigned& height) {
  std::vector<unsigned char> image;  // the raw pixels
  width = 0, height = 0;

  // decode
  unsigned error = lodepng::decode(image, width, height, filename);

  // if there's an error, display it
  if (error)
    std::cerr << "decoder error " << error << ": " << lodepng_error_text(error)
              << std::endl;

  Vector result(width * height);
  for (unsigned i = 0; i < width * height; ++i) {
    result[i] = 1.0 / 3 * (image[4 * i] + image[4 * i + 1] + image[4 * i + 2]);
  }

  return result;
}
