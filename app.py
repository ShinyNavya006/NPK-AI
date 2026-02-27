from flask import Flask, render_template, request
from fertilizer_engine import recommend_fertilizer

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            data = {
                "soil_color": request.form.get("soil_color"),
                "crop": request.form.get("crop"),
                "nitrogen": float(request.form.get("nitrogen", 0)),
                "phosphorus": float(request.form.get("phosphorus", 0)),
                "potassium": float(request.form.get("potassium", 0))
            }

            result = recommend_fertilizer(data)

        except Exception as e:
            result = {
                "fertilizer": "Error",
                "confidence": 0,
                "quantity": 0,
                "cost_per_kg": 0,
                "total_cost": 0,
                "nutrients": {"N": 0, "P": 0, "K": 0},
                "deficiency": {"N": 0, "P": 0, "K": 0},
                "error": str(e)
            }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
