require "csv"
require "shainet"
require "./helpers.cr"

age_model : SHAInet::Network = SHAInet::Network.new
age_model.load_from_file("./model/age.nn")

# data structures to hold the input and results
inputs = Array(Array(Float64)).new
outputs = Array(Array(Float64)).new
ids = Array(Int32).new

# read the file
raw = File.read("./data/test.csv")
csv = CSV.new(raw, headers: true)

# load the data structures
while (csv.next)
  row_arr = Array(Float64).new
  row_arr << salutations(csv.row["Name"]).to_f64
  row_arr << csv.row["Pclass"].to_f64
  row_arr << (csv.row["Sex"] == "male" ? 0_f64 : 1_f64)
  row_arr << age(cvs.row).to_f64
  row_arr << csv.row["SibSp"].to_f64
  row_arr << csv.row["Parch"].to_f64
  row_arr << csv.row["Fare"].to_f64
  row_arr << embarked(csv.row["Embarked"]).to_f64
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

# output the results
File.write("results.csv", result)
