def salutations(name)
  salutations = [
    ["Miss", "Ms"],
    ["Mrs"],
    ["Countess", "Lady"],
    ["Master"],
    ["Mr"],
    ["Capt", "Col", "Dr", "Major", "Rev", "Sir"],
  ]

  results = name.scan(/ ([A-Za-z]+)\./)
  if result = results[0]
    index = salutations.index { |sal| sal.includes? result[1] }
    index || 0
  else
    0
  end
end

def embarked(name)
  index = ["S", "C", "Q"].index(name)
  index || 0
end

def age(model, row)
  age_str = row["Age"]
  return age_str.to_f64 unless age_str == ""

  pred_age_params = Array(Float64).new
  pred_age_params << (salutations(row["Name"]) / 6_f64)
  pred_age_params << (row["Pclass"].to_i / 3_f64)
  pred_age_params << (row["Sex"] == "male" ? 0_f64 : 1_f64)
  pred_age_params << (row["SibSp"].to_i / 8_f64)
  pred_age_params << (row["Parch"].to_i / 6_f64)
  pred_age_params << (row["Fare"].to_f / 512_f64)
  pred_age_params << (embarked(row["Embarked"]) / 3_f64)
  pred_age = model.run(pred_age_params)

  max, idx = 0, 0
  pred_age.each_with_index do |r, i|
    if r > max
      max = r
      idx = i
    end
  end

  age = ((idx * 10) + 5)
  puts "predicted: #{age}"
  age.to_f64
end
