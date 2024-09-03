import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

class LogPage extends StatefulWidget {
  const LogPage({super.key});

  @override
  _LogPageState createState() => _LogPageState();
}

class _LogPageState extends State<LogPage> {
  List<Map<String, String>> _logs = [];

  @override
  void initState() {
    super.initState();
    _loadLogs();
  }

  Future<void> _loadLogs() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String? logsJson = prefs.getString('logs');

    if (logsJson != null) {
      List<dynamic> logsList = jsonDecode(logsJson);
      setState(() {
        _logs = logsList.map((log) => Map<String, String>.from(log)).toList();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Log Absensi'),
      ),
      body: ListView.builder(
        itemCount: _logs.length,
        itemBuilder: (context, index) {
          return ListTile(
            title: Text(_logs[index]['name'] ?? ''),
            subtitle: Text(_logs[index]['timestamp'] ?? ''),
          );
        },
      ),
    );
  }
}
