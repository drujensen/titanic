require "csv"
require "shainet"
require "./helpers.cr"

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

# load the data structures
while (csv.next)
  row_arr = Array(Float64).new
  row_arr << (salutations(csv.row["Name"]) / 6_f64)
  row_arr << csv.row["Pclass"].to_f64
  row_arr << (csv.row["Sex"] == "male" ? 0_f64 : 1_f64)
  row_arr << age(cvs.row).to_f64
  row_arr << csv.row["SibSp"].to_f64
  row_arr << csv.row["Parch"].to_f64
  row_arr << csv.row["Fare"].to_f64
  row_arr << (embarked(csv.row["Embarked"]) / 3_f64)
  inputs << row_arr
  outputs << outcome[csv.row["Survived"]]
end

# normalize the data
normalized = SHAInet::TrainingData.new(inputs, outputs)
normalized.normalize_min_max

# create a network
model : SHAInet::Network = SHAInet::Network.new
model.add_layer(:input, 8, :memory, SHAInet.sigmoid)
model.add_layer(:hidden, 6, :memory, SHAInet.sigmoid)
model.add_layer(:hidden, 1, :eraser, SHAInet.sigmoid)
model.add_layer(:output, 2, :memory, SHAInet.sigmoid)

# connect layers
model.connect_ltl(model.input_layers[0], model.hidden_layers[0], :full)
model.connect_ltl(model.input_layers[0], model.hidden_layers[1], :full)

model.connect_ltl(model.hidden_layers[0], model.output_layers[0], :full)
model.connect_ltl(model.hidden_layers[1], model.output_layers[0], :full)

# optimization settings
model.learning_rate = 0.001
model.momentum = 0.001

# train the network
model.train_batch(normalized.data.shuffle, :adam, :c_ent, epoch = 30000, threshold = 0.0000001, log = 1000, batch_size = 50)
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
