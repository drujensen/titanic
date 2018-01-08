require "csv"

class Passenger
  property id : Int32 = 0
  property survived : Bool = false
  property p_class : Int32 = 0
  property name : String = ""
  property salutation : Int32 = 0
  property sex : String = ""
  property age : Int32 = -1
  property sib_sp : Int32 = 0
  property parch : Int32 = 0
  property ticket : String = ""
  property fare : Float32 = 0_f32
  property cabin : String = ""
  property embarked : Int32 = 0

  def self.load_from_csv(file_name)
    results = [] of Passenger
    raw = File.read(file_name)
    csv = CSV.new(raw, headers: true)
    while (csv.next)
      passenger = Passenger.new
      passenger.id = csv.row["PassengerId"].to_i
      passenger.survived = csv.row["Survived"] == "1"
      passenger.name = csv.row["Name"]
      passenger.salutation = self.salutation_index(csv.row["Name"])
      passenger.p_class = csv.row["Pclass"].to_i
      passenger.sex = csv.row["Sex"]
      passenger.age = csv.row["Age"] == "" ? -1 : csv.row["Age"].split(".")[0].to_i
      passenger.sib_sp = csv.row["SibSp"].to_i
      passenger.parch = csv.row["Parch"].to_i
      passenger.ticket = csv.row["Ticket"]
      passenger.fare = csv.row["Fare"].to_f32
      passenger.cabin = csv.row["Cabin"]
      passenger.embarked = self.embarked_index(csv.row["Embarked"])
      results << passenger
    end
    results
  end

  def calc_age(model)
    pred_age_params = Array(Float64).new
    pred_age_params << (@salutation / 6_f64)
    pred_age_params << (@p_class.to_i / 3_f64)
    pred_age_params << (@sex == "male" ? 0_f64 : 1_f64)
    pred_age_params << (@sib_sp / 8_f64)
    pred_age_params << (@parch / 6_f64)
    pred_age_params << (@fare.to_f64 / 512_f64)
    pred_age_params << (@embarked / 3_f64)
    pred_age = model.run(pred_age_params)

    max, age = 0, 0
    pred_age.each_with_index do |r, i|
      if r > max
        max = r
        age = i
      end
    end

    @age = age
  end

  def self.salutation_index(name)
    salutations = [
      ["Miss", "Ms"],
      ["Master"],
      ["Mrs"],
      ["Mr"],
      ["Countess", "Lady"],
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

  def self.embarked_index(name)
    index = ["S", "C", "Q"].index(name)
    index || 0
  end
end
