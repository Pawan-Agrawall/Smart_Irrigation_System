from flask import Flask, request, jsonify

app = Flask(__name__)

latest_data = { "temperature": None, "humidity": None, "soil": None }

@app.route('/data', methods=['POST'])
def read_data():
    global latest_data
    latest_data = request.get_json()

    print("\n--- New Sensor Data Received ---")
    print(latest_data)
    print("--------------------------------")

    return jsonify({"status": "received"}), 200


@app.route('/latest', methods=['GET'])
def get_latest():
    return jsonify(latest_data), 200


if __name__ == "__main__":
    print("Python server running on port 5000...")
    app.run(host="0.0.0.0", port=5000)
