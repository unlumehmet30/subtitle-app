import 'dart:io';

import 'package:dio/dio.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

class VideoUploadScreen extends StatefulWidget {
  const VideoUploadScreen({super.key});

  @override
  State<VideoUploadScreen> createState() => _VideoUploadScreenState();
}

class _VideoUploadScreenState extends State<VideoUploadScreen> {
  String? _status;
  String? _srtDownloadUrl;
  String? _srtFilePath;

  Future<void> _uploadVideo() async {
    setState(() {
      _status = null;
      _srtDownloadUrl = null;
      _srtFilePath = null;
    });

    final picked = await FilePicker.platform.pickFiles(type: FileType.video);
    if (picked == null || picked.files.isEmpty) {
      setState(() => _status = "Video seçilmedi.");
      return;
    }

    final file = picked.files.first;
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(file.path!, filename: file.name),
    });

    final dio = Dio();

    try {
      final response = await dio.post(
        'http://10.0.2.2:8000/upload', // Local sunucu IP
        data: formData,
        options: Options(contentType: 'multipart/form-data'),
      );

      final downloadPath = response.data['srt_file'];
      setState(() {
        _status = "Video yüklendi. SRT hazır.";
        _srtDownloadUrl = "http://10.0.2.2:8000$downloadPath";
      });
    } catch (e) {
      setState(() => _status = "Yükleme hatası: $e");
    }
  }

  Future<void> _downloadSRT() async {
    if (_srtDownloadUrl == null) return;

    final status = await Permission.storage.request();
    if (!status.isGranted) {
      setState(() => _status = "İzin verilmedi.");
      return;
    }

    final dir = await getExternalStorageDirectory();
    final filePath = "${dir!.path}/subtitle_${DateTime.now().millisecondsSinceEpoch}.srt";

    try {
      await Dio().download(_srtDownloadUrl!, filePath);
      setState(() {
        _srtFilePath = filePath;
        _status = "SRT indirildi: $filePath";
      });
    } catch (e) {
      setState(() => _status = "İndirme hatası: $e");
    }
  }

  Widget _buildSrtPreview() {
    if (_srtFilePath == null) return const SizedBox.shrink();

    return FutureBuilder<String>(
      future: File(_srtFilePath!).readAsString(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const CircularProgressIndicator();
        }
        if (snapshot.hasError) return Text("Okuma hatası: ${snapshot.error}");
        return SingleChildScrollView(
          padding: const EdgeInsets.all(12),
          child: Text(snapshot.data ?? "", style: const TextStyle(fontSize: 14)),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Video Altyazı Yükleyici')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            ElevatedButton(
              onPressed: _uploadVideo,
              child: const Text('🎥 Video Seç ve Gönder'),
            ),
            const SizedBox(height: 16),
            if (_srtDownloadUrl != null)
              ElevatedButton(
                onPressed: _downloadSRT,
                child: const Text("⬇️ SRT Dosyasını İndir"),
              ),
            const SizedBox(height: 16),
            if (_status != null) Text(_status!, textAlign: TextAlign.center),
            const SizedBox(height: 16),
            Expanded(child: _buildSrtPreview()),
          ],
        ),
      ),
    );
  }
}

