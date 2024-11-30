def calculate_probability(data, total_yes, total_no, conditions):
    total = total_yes + total_no
    p_yes = total_yes / total
    p_no = total_no / total

    def get_conditional_probability(feature, value, outcome):
        try:
            return data[feature][value][outcome] / (total_yes if outcome == "Yes" else total_no)
        except KeyError:
            raise ValueError(f"Invalid value '{value}' for feature '{feature}'.")

    p_rain_yes = get_conditional_probability("Outlook", conditions["Outlook"], "Yes")
    p_rain_no = get_conditional_probability("Outlook", conditions["Outlook"], "No")

    p_humidity_yes = get_conditional_probability("Humidity", conditions["Humidity"], "Yes")
    p_humidity_no = get_conditional_probability("Humidity", conditions["Humidity"], "No")

    p_wind_yes = get_conditional_probability("Wind", conditions["Wind"], "Yes")
    p_wind_no = get_conditional_probability("Wind", conditions["Wind"], "No")

    p_yes_given_conditions = p_rain_yes * p_humidity_yes * p_wind_yes * p_yes
    p_no_given_conditions = p_rain_no * p_humidity_no * p_wind_no * p_no

    total_probability = p_yes_given_conditions + p_no_given_conditions
    p_yes_final = p_yes_given_conditions / total_probability
    p_no_final = p_no_given_conditions / total_probability

    return p_yes_final, p_no_final

data = {
    "Outlook": {
        "rainy": {"Yes": 2, "No": 3},
        "sunny": {"Yes": 3, "No": 2},
        "overcast": {"Yes": 4, "No": 0}
    },
    "Humidity": {
        "high": {"Yes": 3, "No": 4},
        "normal": {"Yes": 6, "No": 1}
    },
    "Wind": {
        "weak": {"Yes": 6, "No": 2},
        "strong": {"Yes": 3, "No": 3}
    }
}

total_yes = 9
total_no = 5

test_conditions = {
    "Outlook": "overcast",
    "Humidity": "high",
    "Wind": "strong"
}

try:
    p_yes_final, p_no_final = calculate_probability(data, total_yes, total_no, test_conditions)
    print(f"Conditions: {test_conditions}")
    print(f"Probability that the match will happen (Yes): {p_yes_final:.2%}")
    print(f"Probability that the match will not happen (No): {p_no_final:.2%}")
except ValueError as e:
    print(f"Error: {e}")
