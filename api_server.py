#!/usr/bin/env python3
"""
Simple API server for MuseTalk lip sync service
"""

from flask import Flask, request, jsonify, send_file
import os
import tempfile
from musetalk_wrapper import run_musetalk

app = Flask(__name__)

# Path to the avatar video
AVATAR_VIDEO_PATH = "/app/assets/avatar_video.mp4"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "musetalk"})

@app.route('/lipsync', methods=['POST'])
def create_lipsync():
    """Create lip sync video from audio"""
    try:
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            audio_file.save(temp_audio.name)
            audio_path = temp_audio.name
        
        # Create output file path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
            output_path = temp_output.name
        
        # Run MuseTalk
        result = run_musetalk(
            audio_path=audio_path,
            image_path=AVATAR_VIDEO_PATH,  # Using our avatar video
            output_path=output_path
        )
        
        # Clean up temp audio file
        os.unlink(audio_path)
        
        if result.get("success"):
            # Return the video file
            return send_file(
                result.get("output"),
                as_attachment=True,
                download_name="lipsync_result.mp4",
                mimetype="video/mp4"
            )
        else:
            return jsonify({
                "error": "MuseTalk processing failed",
                "details": result.get("error", "Unknown error")
            }), 500
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    finally:
        # Clean up temp files
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.unlink(audio_path)
            if 'output_path' in locals() and os.path.exists(output_path):
                os.unlink(output_path)
        except:
            pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
