require "csv"
require "shainet"

raw = File.read("./data/train.csv")
csv = CSV.new(raw, headers: true)

headers = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]

inputs = Array(Array(Float64)).new
outputs = Array(Array(Float64)).new
actual = Array(Int32).new

outcome = Hash(Int32, Array(Float64)).new
(0..99).each do |idx|
  outcome[idx] = Array(Float64).new
  (0..99).each do |pos|
    outcome[idx] << (pos == idx ? 1_f64 : 0_f64)
  end
end

while (csv.next)
  next if csv.row["Age"] == ""
  row_arr = Array(Float64).new
  row_arr << csv.row["Survived"].to_f64
  row_arr << csv.row["Pclass"].to_f64
  row_arr << (csv.row["Sex"] == "male" ? 0_f64 : 1_f64)
  row_arr << csv.row["SibSp"].to_f64
  row_arr << csv.row["Parch"].to_f64
  row_arr << csv.row["Fare"].to_f64
  row_arr << (["", "S", "C", "Q"].index(csv.row["Embarked"]).not_nil!.to_f64)
  age = csv.row["Age"].split(".")[0].to_i
  inputs << row_arr
  outputs << outcome[age]
  actual << age
end

normalized = SHAInet::TrainingData.new(inputs, outputs)
normalized.normalize_min_max

# create a network
model : SHAInet::Network = SHAInet::Network.new
model.load_from_file("./model/age.nn")

t = f = 0

# determine accuracy
normalized.normalized_inputs.each_with_index do |test, idx|
  results = model.run(test)
  max, max_idx = 0, 0
  results.each_with_index do |r, i|
    if r > max
      max = r
      max_idx = i
    end
  end
  puts "pred: #{max_idx} actual: #{actual[idx]}"
  if max_idx == actual[idx]
    t += 1
  else
    f += 1
  end
end

puts "Training size: #{outputs.size}"
puts "----------------------"
puts "T: #{t} | F: #{f}"
puts "----------------------"
puts "Accuracy: #{t / outputs.size.to_f}"
