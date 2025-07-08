import 'package:flutter/material.dart';
import 'screens/video_upload_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Subtitle App',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const VideoUploadScreen(),
    );
  }
}


