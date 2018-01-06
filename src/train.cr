require "csv"
require "shainet"

# train the network
age_model : SHAInet::Network = SHAInet::Network.new
age_model.load_from_file("./model/age.nn")

outcome = {
  "0" => [1_f64, 0_f64],
  "1" => [0_f64, 1_f64],
}

# data structures to hold the input and results
inputs = Array(Array(Float64)).new
outputs = Array(Array(Float64)).new

# read the file
raw = File.read("./data/train.csv")
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
    pred_age_params << csv.row["Survived"].to_f64
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
  row_arr << csv.row["Fare"].to_f64
  row_arr << (["", "S", "C", "Q"].index(csv.row["Embarked"]).not_nil!.to_f64)
  
  inputs << row_arr
  outputs << outcome[csv.row["Survived"]]
end

# normalize the data
normalized = SHAInet::TrainingData.new(inputs, outputs)
normalized.normalize_min_max

# create a network
model : SHAInet::Network = SHAInet::Network.new
model.add_layer(:input, 7, "memory", SHAInet.sigmoid)
model.add_layer(:hidden, 14, "memory", SHAInet.sigmoid)
model.add_layer(:output, 2, "memory", SHAInet.sigmoid)
model.fully_connect

# params for sgdm
model.learning_rate = 0.001
model.momentum = 0.001

# train the network
model.train(normalized.data.shuffle, :sgdm, :mse, epoch = 30000, threshold = -1.0, log = 100)
model.save_to_file("./model/titanic.nn")

tn = tp = fn = fp = 0

# determine accuracy
normalized.normalized_inputs.each_with_index do |test, idx|
  results = model.run(test)
  if results[0] < 0.5
    if outputs[idx][0] == 0.0
      tn += 1
    else
      fn += 1
    end
  else
    if outputs[idx][0] == 0.0
      fp += 1
    else
      tp += 1
    end
  end
end

puts "Training size: #{outputs.size}"
puts "----------------------"
puts "TN: #{tn} | FP: #{fp}"
puts "----------------------"
puts "FN: #{fn} | TP: #{tp}"
puts "----------------------"
puts "Accuracy: #{(tn + tp) / outputs.size.to_f}"


