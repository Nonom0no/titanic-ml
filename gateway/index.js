require("dotenv").config();
const express = require("express");
const jwt = require("jsonwebtoken");

const app = express();
app.use(express.json());

// Login endpoint Express (terpisah dari ML JWT)
app.post("/auth/login", (req, res) => {
  const { username, password } = req.body;
  if (!username || password !== "alitkicau") {
    return res.status(401).json({ error: "Kredensial salah" });
  }
  const token = jwt.sign({ sub: username }, process.env.JWT_SECRET, { expiresIn: "1h" });
  res.json({ access_token: token, token_type: "bearer" });
});

app.use("/api", require("./routes/predict"));

app.listen(process.env.PORT || 3000, () => {
  console.log(`Express gateway berjalan di port ${process.env.PORT || 3000}`);
});