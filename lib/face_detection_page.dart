import 'dart:developer' as dev;
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:geolocator/geolocator.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:logging/logging.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:math' as math;  // Perubahan ini dilakukan untuk mengimpor paket matematika

class FacePainter extends CustomPainter {
  final List<Face> faces;

  FacePainter(this.faces);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.transparent // Mengatur warna menjadi transparan
      ..strokeWidth = 0.0          // Mengatur lebar garis menjadi 0
      ..style = PaintingStyle.fill; // Mengubah style menjadi fill

    for (Face face in faces) {
      canvas.drawRect(face.boundingBox, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
class FaceDetectionPage extends StatefulWidget {
  final List<CameraDescription> cameras;

  const FaceDetectionPage({super.key, required this.cameras});

  @override
  _FaceDetectionPageState createState() => _FaceDetectionPageState();
}

class _FaceDetectionPageState extends State<FaceDetectionPage> {
  late CameraController _controller;
  late Interpreter _interpreter;
  static const String modelFile = "assets/model.tflite";
  bool _isDetecting = false;
  bool _retryVisible = false;
  final double _targetLatitude = 37.4219983;
  final double _targetLongitude = -122.084;
  final double _locationRadius = 100;
  String _recognizedName = '';
  Rect? _recognizedBox;
  List<String> labels = [];
  final log = Logger('FaceDetectionPage');
  bool isControllerInitialized = false;
  img.Image? croppedFace;
  late FaceDetector faceDetector;
  List<Face> faces = [];
  bool _isDetected = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    loadModel();
    loadLabels();
    final options = FaceDetectorOptions(
      performanceMode: FaceDetectorMode.accurate,
    );
    faceDetector = FaceDetector(options: options);
  }

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        modelFile,
        options: InterpreterOptions()..threads = 4,
      );

      _interpreter.allocateTensors();
      dev.log("Model loaded successfully");
    } catch (e) {
      dev.log("Error while creating interpreter: $e");
    }
  }

  Future<void> loadLabels() async {
    try {
      final labelsString = await rootBundle.loadString('assets/labels.txt');
      labels = labelsString.split('\n');
      dev.log("Labels loaded successfully: $labels");
    } catch (e) {
      dev.log("Error loading labels: $e");
    }
  }

  Future<void> _initializeCamera() async {
    try {
      final frontCamera = widget.cameras.firstWhere(
          (camera) => camera.lensDirection == CameraLensDirection.front);

      _controller = CameraController(frontCamera, ResolutionPreset.high);
      await _controller.initialize();
      isControllerInitialized = true;
      setState(() {});
      dev.log("Camera initialized successfully");
    } catch (e) {
      dev.log("Error initializing camera: $e");
    }
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

      croppedFace = img.copyCrop(capturedImage,
          x: left.toInt(),
          y: top.toInt(),
          width: width.toInt(),
          height: height.toInt());

      await predict(croppedFace!);

      setState(() {
        _recognizedBox = faceRect;
      });

      // Detect multiple faces using IoU
      for (int i = 0; i < faces.length - 1; i++) {
        for (int j = i + 1; j < faces.length; j++) {
          double iou = calculateIoU(faces[i].boundingBox, faces[j].boundingBox);
          if (iou > 0.5) {
            // Handle overlapping faces here (you might want to merge or keep the one with a higher confidence)
            dev.log("Overlapping faces detected with IoU: $iou");
          }
        }
      }
    }
  }

  double calculateIoU(Rect boxA, Rect boxB) {
    final double xA = math.max(boxA.left, boxB.left);
    final double yA = math.max(boxA.top, boxB.top);
    final double xB = math.min(boxA.right, boxB.right);
    final double yB = math.min(boxA.bottom, boxB.bottom);

    final double interArea = math.max(0, xB - xA) * math.max(0, yB - yA);

    final double boxAArea = (boxA.right - boxA.left) * (boxA.bottom - boxA.top);
    final double boxBArea = (boxB.right - boxB.left) * (boxB.bottom - boxB.top);

    final double iou = interArea / (boxAArea + boxBArea - interArea);
    return iou;
  }

  Future<File> removeRotation(File inputImage) async {
    final img.Image? capturedImage =
        img.decodeImage(await File(inputImage.path).readAsBytes());
    final img.Image orientedImage = img.bakeOrientation(capturedImage!);
    return await File(inputImage.path)
        .writeAsBytes(img.encodeJpg(orientedImage));
  }

  List<dynamic> imageToArray(img.Image inputImage, int width, int height) {
    img.Image resizedImage =
        img.copyResize(inputImage, width: width, height: height);
    List<double> flattenedList = resizedImage.data!
        .expand((channel) =>
            [channel.r.toDouble(), channel.g.toDouble(), channel.b.toDouble()])
        .toList();
    Float32List float32Array = Float32List.fromList(flattenedList);
    Float32List reshapedArray = Float32List(1 * height * width * 3);
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index = h * width + w;
        reshapedArray[index * 3] = (float32Array[index * 3] - 127.5) / 127.5;
        reshapedArray[index * 3 + 1] =
            (float32Array[index * 3 + 1] - 127.5) / 127.5;
        reshapedArray[index * 3 + 2] =
            (float32Array[index * 3 + 2] - 127.5) / 127.5;
      }
    }
    return reshapedArray.reshape([1, height, width, 3]);
  }

  Future<void> predict(img.Image image) async {
    img.Image resizedImage = img.copyResize(image, width: 112, height: 112);

    // Convert the resized image to a 1D Float32List.
    var inputBytes = imageToArray(resizedImage, 112, 112);

    // Create output tensor
    var output = List.generate(1, (_) => List.filled(71, 0.0)).reshape([1, 71]);

    // Run inference
    _interpreter.run(inputBytes, output);

    final predictionResult = output[0] as List<double>;
    double maxElement = predictionResult.reduce(
      (double maxElement, double element) =>
          element > maxElement ? element : maxElement,
    );

    final index = predictionResult.indexOf(maxElement);
    setState(() {
      _recognizedName = labels[index];
      _isDetected = true;
    });

    _saveLog(_recognizedName);
  }

  Future<void> _saveLog(String name) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String timestamp = DateTime.now().toString();
    List<Map<String, String>> logs = [];
    String? existingLogs = prefs.getString('logs');
    if (existingLogs != null) {
      List<dynamic> decodedLogs = jsonDecode(existingLogs);
      logs = decodedLogs.map((e) => Map<String, String>.from(e)).toList();
    }
    logs.add({'name': name, 'timestamp': timestamp});
    prefs.setString('logs', jsonEncode(logs));
  }

  Future<bool> _isUserInLocation() async {
    Position position = await Geolocator.getCurrentPosition(
      desiredAccuracy: LocationAccuracy.high,
    );
    double distance = Geolocator.distanceBetween(
      _targetLatitude,
      _targetLongitude,
      position.latitude,
      position.longitude,
    );
    return distance <= _locationRadius;
  }

  void _showAlert(String title, String message) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text(title),
          content: Text(message),
          actions: <Widget>[
            TextButton(
              child: const Text('OK'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text('Absensi dengan Deteksi Wajah'),
      ),
      body: Stack(
        children: [
          Column(
            children: [
              SizedBox(
                width: MediaQuery.of(context).size.width,
                height: MediaQuery.of(context).size.height * 0.7,
                child: OverflowBox(
                  alignment: Alignment.center,
                  child: FittedBox(
                    fit: BoxFit.fitWidth,
                    child: SizedBox(
                        width: MediaQuery.of(context).size.width,
                        height: MediaQuery.of(context).size.width *
                            _controller.value.aspectRatio,
                        child: isControllerInitialized
                            ? Stack(
                                children: [
                                  CameraPreview(_controller),
                                  CustomPaint(
                                    painter: FacePainter(faces),
                                  ),
                                ],
                              )
                            : null),
                  ),
                ),
              ),
              ElevatedButton(
                onPressed: _isDetecting
                    ? null
                    : () async {
                        setState(() {
                          _isDetecting = true;
                        });

                        if (!_controller.value.isInitialized) {
                          dev.log("Camera is not initialized");
                          setState(() {
                            _isDetecting = false;
                          });
                          return;
                        }

                        // Try to capture an image
                        XFile imageFile = await _controller.takePicture();
                        final img.Image? imageData = img.decodeImage(
                            File(imageFile.path).readAsBytesSync());

                        if (imageData == null) {
                          dev.log("Failed to decode image");
                          setState(() {
                            _isDetecting = false;
                            _retryVisible = true;
                          });
                          return;
                        }

                        await doFaceDetection(File(imageFile.path));

                        await predict(imageData);

                        setState(() {
                          _isDetecting = false;
                        });
                      },
                child: const Text('Ambil Foto dan Absensi'),
              ),
              if (_retryVisible)
                ElevatedButton(
                  onPressed: () {
                    setState(() {
                      _retryVisible = false;
                    });
                  },
                  child: const Text('Coba Lagi'),
                ),
            ],
          ),
          Visibility(
            visible: _isDetected,
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Container(
                color: Colors.white,
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      'Wajah Terdeteksi: $_recognizedName',
                      style: const TextStyle(
                        fontSize: 24.0,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16.0),
                    ElevatedButton(
                      onPressed: () {
                        setState(() {
                          _isDetected = false;
                        });
                      },
                      child: const Text('Absen'),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
