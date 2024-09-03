import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:nb_utils/nb_utils.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class MLService {
  late Interpreter interpreter;
  late InterpreterOptions _interpreterOptions;
  late FaceDetector faceDetector;
  List<Face> faces = [];
  File? image;
  img.Image? capturedImage;
  img.Image? cropedFace;
  late List<String> labels;

  MLService({int? numThreads}) {
    _interpreterOptions = InterpreterOptions();
    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }
    final options = FaceDetectorOptions(
      performanceMode: FaceDetectorMode.accurate,
    );
    faceDetector = FaceDetector(options: options);
    initialize();
  }

  Future<void> loadModel({String? modelName, int? numThreads}) async {
    interpreter = await Interpreter.fromAsset('assets/mobilefacenet.tflite');
  }

  List<dynamic> imageToArray(img.Image inputImage, int width, int height) {
    img.Image resizedImage =
        img.copyResize(inputImage, width: width, height: height);
    List<double> flattenedList = resizedImage.data!
        .expand((channel) => [channel.r, channel.g, channel.b])
        .map((value) => value.toDouble())
        .toList();
    Float32List float32Array = Float32List.fromList(flattenedList);
    int channels = 3;
    Float32List reshapedArray = Float32List(1 * height * width * channels);
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int index = c * height * width + h * width + w;
          reshapedArray[index] =
              (float32Array[c * height * width + h * width + w] - 127.5) /
                  127.5;
        }
      }
    }
    return reshapedArray.reshape([1, 112, 112, 3]);
  }

  Future<List<double>> getEmbeddings(img.Image image) async {
    List input = imageToArray(image, 112, 112);
    List output = List.filled(1 * 192, 0).reshape([1, 192]);
    interpreter.run(input, output);
    interpreter.close();
    return List.from(output.reshape([192]));
  }

  Future<bool> compareFaces(
      img.Image capturedImage, List<double> userEmbeddings) async {
    List<double> embeddings = await getEmbeddings(capturedImage);
    double distance = euclideanDistance(embeddings, userEmbeddings);
    double threshold = 0.8;
    if (distance <= threshold) {
      return true;
    } else {
      return false;
    }
  }

  double euclideanDistance(List<double> embeddings1, List<double> embeddings2) {
    double sum = 0.0;
    for (int i = 0; i < embeddings1.length; i++) {
      double diff = embeddings1[i] - embeddings2[i];
      sum += pow(diff, 2);
    }
    return sqrt(sum);
  }

  Future<void> doFaceDetection(File image) async {
    File? image0 = await removeRotation(image);
    final Uint8List bytes = await image0.readAsBytes();
    final capturedImage = img.decodeImage(bytes)!;

    InputImage inputImage = InputImage.fromFile(image0);

    faces = await faceDetector.processImage(inputImage);

    if (faces.isEmpty) return;

    for (Face face in faces) {
      final Rect faceRect = face.boundingBox;
      num left = faceRect.left < 0 ? 0 : faceRect.left;
      num top = faceRect.top < 0 ? 0 : faceRect.top;
      num right = faceRect.right > capturedImage.width
          ? capturedImage.width - 1
          : faceRect.right;
      num bottom = faceRect.bottom > capturedImage.height
          ? capturedImage.height - 1
          : faceRect.bottom;
      num width = right - left;
      num height = bottom - top;

      cropedFace = img.copyCrop(capturedImage,
          x: left.toInt(),
          y: top.toInt(),
          width: width.toInt(),
          height: height.toInt());

      return;
    }
  }

  Future<File> removeRotation(File inputImage) async {
    final img.Image? capturedImage =
        img.decodeImage(await File(inputImage.path).readAsBytes());
    final img.Image orientedImage = img.bakeOrientation(capturedImage!);
    return await File(inputImage.path)
        .writeAsBytes(img.encodeJpg(orientedImage));
  }

  void clear() {
    cropedFace = null;
    faces.clear();
  }
}
