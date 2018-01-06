require "csv"
require "shainet"

age_model : SHAInet::Network = SHAInet::Network.new
age_model.load_from_file("./model/age.nn")

# data structures to hold the input and results
inputs = Array(Array(Float64)).new
outputs = Array(Array(Float64)).new
ids = Array(Int32).new

# read the file
raw = File.read("./data/test.csv")
csv = CSV.new(raw, headers: true)

# we don't want these columns so we won't load them
headers = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# load the data structures
while (csv.next)
  row_arr = Array(Float64).new
  row_arr << csv.row["Pclass"].to_f64
  row_arr << (csv.row["Sex"] == "male" ? 0_f64 : 1_f64)

  if csv.row["Age"] == ""
    pred_age_params = Array(Float64).new
    pred_age_params << 1_f64
    pred_age_params << csv.row["Pclass"].to_f64
    pred_age_params << (csv.row["Sex"] == "male" ? 0_f64 : 1_f64)
    pred_age_params << csv.row["SibSp"].to_f64
    pred_age_params << csv.row["Parch"].to_f64
    pred_age_params << csv.row["Fare"].to_f64
    pred_age_params << (["", "S", "C", "Q"].index(csv.row["Embarked"]).not_nil!.to_f64)
    pred_age = age_model.run(pred_age_params)
    max, age = 0, 0
    pred_age.each_with_index do |r, i|
      if r > max
        max = r
        age = i
      end
    end
    puts "pred age: #{age}"
    row_arr << age.to_f64
  else
    row_arr << csv.row["Age"].split(".")[0].to_f64
  end

  row_arr << csv.row["SibSp"].to_f64
  row_arr << csv.row["Parch"].to_f64
  puts "Fare #{csv.row["Fare"]}"
  row_arr << (csv.row["Fare"] != "" ? csv.row["Fare"].to_f64 : 0_f64)
  row_arr << (["", "S", "C", "Q"].index(csv.row["Embarked"]).not_nil!.to_f64)
  inputs << row_arr
  ids << csv.row["PassengerId"].to_i32
end

# normalize the data
normalized = SHAInet::TrainingData.new(inputs, outputs)
normalized.normalize_min_max

# load model
model : SHAInet::Network = SHAInet::Network.new
model.load_from_file("./model/titanic.nn")

result = CSV.build do |csv|
  csv.row "PassengerId", "Survived"
  # determine accuracy
  normalized.normalized_inputs.each_with_index do |test, idx|
    results = model.run(test)
    if results[0] < 0.5
      csv.row ids[idx], 0
    else
      csv.row ids[idx], 1
    end
  end
end
puts result
