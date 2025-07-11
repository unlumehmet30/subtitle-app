// Gerekli paketler:
// dio, file_picker, video_player, chewie, path_provider, permission_handler

import 'dart:io';
import 'package:chewie/chewie.dart';
import 'package:dio/dio.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

class VideoUploadScreen extends StatefulWidget {
  const VideoUploadScreen({super.key});

  @override
  State<VideoUploadScreen> createState() => _VideoUploadScreenState();
}

class _VideoUploadScreenState extends State<VideoUploadScreen> {
  File? _videoFile;
  File? _existingSrtFile;
  String? _status;
  String? _finalVideoUrl;
  String? _subtitleUrl;
  bool _isProcessing = false;
  double _progress = 0.0;

  VideoPlayerController? _videoController;
  ChewieController? _chewieController;

  final Dio _dio = Dio();
  static const String _baseUrl = "http://localhost:8000"; // Python server URL

  // AI Subtitle Generation seçenekleri
  String _sourceLanguage = "tr";
  String? _targetLanguage;
  String _modelType = "whisper";
  bool _translate = false;
  bool _burnSubtitles = true;
  double _confidenceThreshold = 0.7;

  // İşlem modu
  ProcessingMode _processingMode = ProcessingMode.aiGenerate;

  // Desteklenen diller
  static const Map<String, String> _supportedLanguages = {
    "tr": "Türkçe",
    "en": "İngilizce",
    "es": "İspanyolca",
    "fr": "Fransızca",
    "de": "Almanca",
    "it": "İtalyanca",
    "pt": "Portekizce",
    "ru": "Rusça",
    "ja": "Japonca",
    "ko": "Korece",
    "ar": "Arapça",
    "zh": "Çince"
  };

  @override
  void initState() {
    super.initState();
    _checkServerHealth();
  }

  Future<void> _checkServerHealth() async {
    try {
      final response = await _dio.get('$_baseUrl/health');
      if (response.statusCode == 200) {
        final data = response.data;
        setState(() {
          _status = "Server bağlantısı başarılı ✅\n"
              "Cihaz: ${data['device']}\n"
              "Yüklü modeller: ${data['models_loaded'].join(', ')}";
        });
      }
    } catch (e) {
      setState(() {
        _status = "Server bağlantısı başarısız ❌ ($e)";
      });
    }
  }

  Future<void> _pickVideo() async {
    try {
      final picked = await FilePicker.platform.pickFiles(
        type: FileType.video,
        allowMultiple: false,
      );
      
      if (picked != null && picked.files.isNotEmpty) {
        setState(() {
          _videoFile = File(picked.files.first.path!);
          _status = "Video seçildi: ${picked.files.first.name}";
        });
        await _initializeVideoPlayer();
      }
    } catch (e) {
      setState(() {
        _status = "Video seçme hatası: $e";
      });
    }
  }

  Future<void> _pickSubtitleFile() async {
    try {
      final picked = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['srt'],
        allowMultiple: false,
      );
      
      if (picked != null && picked.files.isNotEmpty) {
        setState(() {
          _existingSrtFile = File(picked.files.first.path!);
          _status = "Altyazı dosyası seçildi: ${picked.files.first.name}";
        });
      }
    } catch (e) {
      setState(() {
        _status = "Altyazı seçme hatası: $e";
      });
    }
  }

  Future<void> _initializeVideoPlayer() async {
    if (_videoFile == null) return;

    try {
      _videoController = VideoPlayerController.file(_videoFile!)
        ..initialize().then((_) {
          _chewieController = ChewieController(
            videoPlayerController: _videoController!,
            autoPlay: false,
            looping: false,
            showControls: true,
            materialProgressColors: ChewieProgressColors(
              playedColor: Colors.blue,
              handleColor: Colors.blueAccent,
              backgroundColor: Colors.grey,
              bufferedColor: Colors.lightBlue,
            ),
            placeholder: Container(
              color: Colors.black,
              child: const Center(
                child: CircularProgressIndicator(),
              ),
            ),
          );
          if (mounted) {
            setState(() {});
          }
        });
    } catch (e) {
      setState(() {
        _status = "Video oynatıcı hatası: $e";
      });
    }
  }

  Future<void> _processVideo() async {
    if (_videoFile == null) {
      setState(() => _status = "Video dosyası seçilmeli");
      return;
    }

    if (_processingMode == ProcessingMode.translateSrt && _existingSrtFile == null) {
      setState(() => _status = "Çeviri için altyazı dosyası seçilmeli");
      return;
    }

    setState(() {
      _isProcessing = true;
      _progress = 0.0;
      _status = "İşlem başlatılıyor...";
    });

    try {
      await _requestPermissions();
      
      if (_processingMode == ProcessingMode.aiGenerate) {
        await _generateAiSubtitles();
      } else {
        await _translateExistingSubtitles();
      }
      
    } catch (e) {
      setState(() {
        _status = "Hata: $e";
        _isProcessing = false;
      });
    }
  }

  Future<void> _generateAiSubtitles() async {
    try {
      setState(() => _status = "AI ile altyazı üretiliyor...");

      FormData formData = FormData.fromMap({
        'video': await MultipartFile.fromFile(
          _videoFile!.path,
          filename: _videoFile!.path.split('/').last,
        ),
        'source_language': _sourceLanguage,
        'target_language': _translate ? _targetLanguage : null,
        'model_type': _modelType,
        'translate': _translate,
        'confidence_threshold': _confidenceThreshold,
        'burn_subtitles': _burnSubtitles,
      });

      final response = await _dio.post(
        '$_baseUrl/generate-ai-subtitles',
        data: formData,
        options: Options(
          contentType: 'multipart/form-data',
          responseType: ResponseType.bytes,
        ),
        onSendProgress: (sent, total) {
          if (total != -1) {
            setState(() {
              _progress = sent / total;
            });
          }
        },
      );

      if (response.statusCode == 200) {
        final contentType = response.headers['content-type']?.first ?? '';
        
        if (contentType.contains('video/')) {
          await _saveProcessedVideo(response.data);
        } else {
          await _saveSubtitleFile(response.data);
        }
      } else {
        throw Exception('Server hatası: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('AI altyazı üretme hatası: $e');
    }
  }

  Future<void> _translateExistingSubtitles() async {
    try {
      setState(() => _status = "Altyazı çevriliyor...");

      FormData formData = FormData.fromMap({
        'srt_file': await MultipartFile.fromFile(
          _existingSrtFile!.path,
          filename: _existingSrtFile!.path.split('/').last,
        ),
        'source_language': _sourceLanguage,
        'target_language': _targetLanguage ?? 'en',
      });

      final response = await _dio.post(
        '$_baseUrl/translate-subtitles',
        data: formData,
        options: Options(
          contentType: 'multipart/form-data',
          responseType: ResponseType.bytes,
        ),
        onSendProgress: (sent, total) {
          if (total != -1) {
            setState(() {
              _progress = sent / total;
            });
          }
        },
      );

      if (response.statusCode == 200) {
        await _saveSubtitleFile(response.data);
      } else {
        throw Exception('Server hatası: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Altyazı çeviri hatası: $e');
    }
  }

  Future<void> _saveProcessedVideo(List<int> videoData) async {
    try {
      final directory = await getExternalStorageDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final filename = 'ai_subtitled_video_$timestamp.mp4';
      final savePath = '${directory!.path}/$filename';

      final file = File(savePath);
      await file.writeAsBytes(videoData);

      setState(() {
        _finalVideoUrl = savePath;
        _status = "Video başarıyla kaydedildi: $filename";
        _isProcessing = false;
        _progress = 1.0;
      });
    } catch (e) {
      setState(() {
        _status = "Video kaydetme hatası: $e";
        _isProcessing = false;
      });
    }
  }

  Future<void> _saveSubtitleFile(List<int> subtitleData) async {
    try {
      final directory = await getExternalStorageDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final filename = 'subtitles_$timestamp.srt';
      final savePath = '${directory!.path}/$filename';

      final file = File(savePath);
      await file.writeAsBytes(subtitleData);

      setState(() {
        _subtitleUrl = savePath;
        _status = "Altyazı dosyası kaydedildi: $filename";
        _isProcessing = false;
        _progress = 1.0;
      });
    } catch (e) {
      setState(() {
        _status = "Altyazı kaydetme hatası: $e";
        _isProcessing = false;
      });
    }
  }

  Future<void> _requestPermissions() async {
    final status = await Permission.storage.request();
    if (status.isDenied) {
      throw Exception('Depolama izni gerekli');
    }
  }

  Future<void> _cleanupServer() async {
    try {
      await _dio.delete('$_baseUrl/cleanup');
      setState(() {
        _status = "Server geçici dosyaları temizlendi";
      });
    } catch (e) {
      setState(() {
        _status = "Temizleme hatası: $e";
      });
    }
  }

  @override
  void dispose() {
    _videoController?.dispose();
    _chewieController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("AI Altyazı Üretici"),
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.cleaning_services),
            onPressed: _cleanupServer,
            tooltip: "Server'ı temizle",
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // İşlem Modu Seçimi
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    const Text("İşlem Modu", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 10),
                    SegmentedButton<ProcessingMode>(
                      segments: const [
                        ButtonSegment(
                          value: ProcessingMode.aiGenerate,
                          label: Text("AI Üret"),
                          icon: Icon(Icons.auto_awesome),
                        ),
                        ButtonSegment(
                          value: ProcessingMode.translateSrt,
                          label: Text("Çevir"),
                          icon: Icon(Icons.translate),
                        ),
                      ],
                      selected: {_processingMode},
                      onSelectionChanged: (Set<ProcessingMode> selected) {
                        setState(() {
                          _processingMode = selected.first;
                        });
                      },
                    ),
                  ],
                ),
              ),
            ),

            // Video Seçimi
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    const Text("Video Seçimi", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 10),
                    ElevatedButton.icon(
                      onPressed: _pickVideo,
                      icon: const Icon(Icons.video_library),
                      label: const Text("Video Seç"),
                      style: ElevatedButton.styleFrom(
                        minimumSize: const Size(double.infinity, 45),
                      ),
                    ),
                    if (_videoFile != null) ...[
                      const SizedBox(height: 10),
                      Text("Seçilen: ${_videoFile!.path.split('/').last}"),
                    ],
                  ],
                ),
              ),
            ),

            // Video Oynatıcı
            if (_videoFile != null && _chewieController != null)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      const Text("Video Önizleme", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 10),
                      AspectRatio(
                        aspectRatio: _videoController!.value.aspectRatio,
                        child: Chewie(controller: _chewieController!),
                      ),
                    ],
                  ),
                ),
              ),

            // Mevcut Altyazı Dosyası (Çeviri modu için)
            if (_processingMode == ProcessingMode.translateSrt)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      const Text("Mevcut Altyazı Dosyası", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 10),
                      ElevatedButton.icon(
                        onPressed: _pickSubtitleFile,
                        icon: const Icon(Icons.subtitles),
                        label: const Text("SRT Dosyası Seç"),
                        style: ElevatedButton.styleFrom(
                          minimumSize: const Size(double.infinity, 45),
                        ),
                      ),
                      if (_existingSrtFile != null) ...[
                        const SizedBox(height: 10),
                        Text("Seçilen: ${_existingSrtFile!.path.split('/').last}"),
                      ],
                    ],
                  ),
                ),
              ),

            // AI Ayarları
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    const Text("AI Ayarları", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 10),
                    
                    // Model Seçimi (sadece AI üretim modu için)
                    if (_processingMode == ProcessingMode.aiGenerate) ...[
                      const Text("AI Modeli:", style: TextStyle(fontWeight: FontWeight.w500)),
                      const SizedBox(height: 5),
                      SegmentedButton<String>(
                        segments: const [
                          ButtonSegment(
                            value: "whisper",
                            label: Text("Whisper"),
                            icon: Icon(Icons.mic),
                          ),
                          ButtonSegment(
                            value: "wav2vec2",
                            label: Text("Wav2Vec2"),
                            icon: Icon(Icons.graphic_eq),
                          ),
                        ],
                        selected: {_modelType},
                        onSelectionChanged: (Set<String> selected) {
                          setState(() {
                            _modelType = selected.first;
                          });
                        },
                      ),
                      const SizedBox(height: 15),
                    ],

                    // Kaynak Dil
                    const Text("Kaynak Dil:", style: TextStyle(fontWeight: FontWeight.w500)),
                    const SizedBox(height: 5),
                    DropdownButton<String>(
                      value: _sourceLanguage,
                      isExpanded: true,
                      items: _supportedLanguages.entries.map((entry) {
                        return DropdownMenuItem(
                          value: entry.key,
                          child: Text(entry.value),
                        );
                      }).toList(),
                      onChanged: (value) {
                        if (value != null) {
                          setState(() {
                            _sourceLanguage = value;
                          });
                        }
                      },
                    ),
                    const SizedBox(height: 15),

                    // Çeviri Seçenekleri
                    SwitchListTile(
                      title: const Text("Çeviri Yap"),
                      subtitle: const Text("Altyazıları başka dile çevir"),
                      value: _translate,
                      onChanged: (value) {
                        setState(() {
                          _translate = value;
                        });
                      },
                    ),

                    if (_translate) ...[
                      const Text("Hedef Dil:", style: TextStyle(fontWeight: FontWeight.w500)),
                      const SizedBox(height: 5),
                      DropdownButton<String>(
                        value: _targetLanguage,
                        hint: const Text("Hedef dil seçin"),
                        isExpanded: true,
                        items: _supportedLanguages.entries.map((entry) {
                          return DropdownMenuItem(
                            value: entry.key,
                            child: Text(entry.value),
                          );
                        }).toList(),
                        onChanged: (value) {
                          setState(() {
                            _targetLanguage = value;
                          });
                        },
                      ),
                      const SizedBox(height: 15),
                    ],

                    // Güven Eşiği (sadece AI üretim modu için)
                    if (_processingMode == ProcessingMode.aiGenerate) ...[
                      const Text("Güven Eşiği:", style: TextStyle(fontWeight: FontWeight.w500)),
                      const SizedBox(height: 5),
                      Slider(
                        value: _confidenceThreshold,
                        min: 0.0,
                        max: 1.0,
                        divisions: 10,
                        label: _confidenceThreshold.toStringAsFixed(1),
                        onChanged: (value) {
                          setState(() {
                            _confidenceThreshold = value;
                          });
                        },
                      ),
                      const SizedBox(height: 15),
                    ],

                    // Çıktı Seçenekleri
                    SwitchListTile(
                      title: const Text("Altyazıları Videoya Göm"),
                      subtitle: const Text("Altyazılar kalıcı olarak videoya işlenir"),
                      value: _burnSubtitles,
                      onChanged: (value) {
                        setState(() {
                          _burnSubtitles = value;
                        });
                      },
                    ),
                  ],
                ),
              ),
            ),

            // İşleme Butonu
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    ElevatedButton.icon(
                      onPressed: _isProcessing ? null : _processVideo,
                      icon: _isProcessing
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.auto_awesome),
                      label: Text(_isProcessing ? "İşleniyor..." : 
                        _processingMode == ProcessingMode.aiGenerate ? "AI ile Altyazı Üret" : "Altyazı Çevir"),
                      style: ElevatedButton.styleFrom(
                        minimumSize: const Size(double.infinity, 50),
                        backgroundColor: Colors.blue,
                        foregroundColor: Colors.white,
                      ),
                    ),
                    if (_isProcessing) ...[
                      const SizedBox(height: 10),
                      LinearProgressIndicator(
                        value: _progress,
                        backgroundColor: Colors.grey[300],
                        valueColor: const AlwaysStoppedAnimation<Color>(Colors.blue),
                      ),
                      const SizedBox(height: 5),
                      Text("${(_progress * 100).toStringAsFixed(1)}%"),
                    ],
                  ],
                ),
              ),
            ),

            // Durum Bilgisi
            if (_status != null)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      const Text("Durum", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 10),
                      Text(
                        _status!,
                        style: TextStyle(
                          color: _status!.contains("Hata") ? Colors.red : Colors.green,
                          fontWeight: FontWeight.w500,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),

            // Sonuç Dosyaları
            if (_finalVideoUrl != null || _subtitleUrl != null)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      const Text("Sonuç Dosyaları", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 10),
                      
                      if (_finalVideoUrl != null) ...[
                        Text(
                          "Video: ${_finalVideoUrl!.split('/').last}",
                          style: const TextStyle(fontSize: 12),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 10),
                        ElevatedButton.icon(
                          onPressed: () => _playProcessedVideo(),
                          icon: const Icon(Icons.play_circle_filled),
                          label: const Text("İşlenmiş Videoyu Oynat"),
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size(double.infinity, 45),
                          ),
                        ),
                      ],
                      
                      if (_subtitleUrl != null) ...[
                        const SizedBox(height: 10),
                        Text(
                          "Altyazı: ${_subtitleUrl!.split('/').last}",
                          style: const TextStyle(fontSize: 12),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 10),
                        ElevatedButton.icon(
                          onPressed: () => _showSubtitleContent(),
                          icon: const Icon(Icons.subtitles),
                          label: const Text("Altyazı İçeriğini Görüntüle"),
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size(double.infinity, 45),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Future<void> _playProcessedVideo() async {
    if (_finalVideoUrl == null) return;

    try {
      final processedFile = File(_finalVideoUrl!);
      if (await processedFile.exists()) {
        final controller = VideoPlayerController.file(processedFile);
        await controller.initialize();

        final chewieController = ChewieController(
          videoPlayerController: controller,
          autoPlay: true,
          looping: false,
          showControls: true,
        );

        if (mounted) {
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (context) => VideoPlayerScreen(
                chewieController: chewieController,
                title: "İşlenmiş Video",
              ),
            ),
          );
        }
      } else {
        setState(() {
          _status = "Video dosyası bulunamadı";
        });
      }
    } catch (e) {
      setState(() {
        _status = "Video oynatma hatası: $e";
      });
    }
  }

  Future<void> _showSubtitleContent() async {
    if (_subtitleUrl == null) return;

    try {
      final subtitleFile = File(_subtitleUrl!);
      if (await subtitleFile.exists()) {
        final content = await subtitleFile.readAsString();
        
        if (mounted) {
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (context) => SubtitleViewerScreen(
                content: content,
                title: "Altyazı İçeriği",
              ),
            ),
          );
        }
      } else {
        setState(() {
          _status = "Altyazı dosyası bulunamadı";
        });
      }
    } catch (e) {
      setState(() {
        _status = "Altyazı okuma hatası: $e";
      });
    }
  }
}

enum ProcessingMode {
  aiGenerate,
  translateSrt,
}

class VideoPlayerScreen extends StatefulWidget {
  final ChewieController chewieController;
  final String title;

  const VideoPlayerScreen({
    super.key,
    required this.chewieController,
    required this.title,
  });

  @override
  State<VideoPlayerScreen> createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  @override
  void dispose() {
    widget.chewieController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
      ),
      backgroundColor: Colors.black,
      body: Center(
        child: Chewie(controller: widget.chewieController),
      ),
    );
  }
}

class SubtitleViewerScreen extends StatelessWidget {
  final String content;
  final String title;

  const SubtitleViewerScreen({
    super.key,
    required this.content,
    required this.title,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
        actions: [
          IconButton(
            icon: const Icon(Icons.copy),
            onPressed: () {
              // Clipboard'a kopyala
              // Clipboard.setData(ClipboardData(text: content));
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text("Altyazı içeriği kopyalandı")),
              );
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Card(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Altyazı İçeriği (.srt)",
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 16),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.grey[100],
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.grey[300]!),
                  ),
                  child: Text(
                    content,
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}