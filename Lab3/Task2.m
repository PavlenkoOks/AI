% Створення нечіткої системи
fuzzy_system = mamfis('Name', 'WaterMixerControl');

% Додавання вхідної змінної для температури
fuzzy_system = addInput(fuzzy_system, [0 100], 'Name', 'Temperature');
fuzzy_system = addMF(fuzzy_system, 'Temperature', 'trapmf', [0 0 20 45], 'Name', 'Cold');
fuzzy_system = addMF(fuzzy_system, 'Temperature', 'trimf', [20 45 75], 'Name', 'Warm');
fuzzy_system = addMF(fuzzy_system, 'Temperature', 'trapmf', [45 75 100 100], 'Name', 'Hot');

% Додавання вхідної змінної для потоку
fuzzy_system = addInput(fuzzy_system, [0 10], 'Name', 'Flow');
fuzzy_system = addMF(fuzzy_system, 'Flow', 'trimf', [0 2 5], 'Name', 'Weak');
fuzzy_system = addMF(fuzzy_system, 'Flow', 'trimf', [2 5 8], 'Name', 'Moderate');
fuzzy_system = addMF(fuzzy_system, 'Flow', 'trapmf', [5 8 10 10], 'Name', 'Strong');

% Додавання вихідної змінної для кута повороту крану гарячої води
fuzzy_system = addOutput(fuzzy_system, [-90 90], 'Name', 'HotWaterValve');
fuzzy_system = addMF(fuzzy_system, 'HotWaterValve', 'trapmf', [-90 -90 -45 0], 'Name', 'TurnLeftLarge');
fuzzy_system = addMF(fuzzy_system, 'HotWaterValve', 'trimf', [-45 0 45], 'Name', 'TurnLeftMedium');
fuzzy_system = addMF(fuzzy_system, 'HotWaterValve', 'trapmf', [0 45 90 90], 'Name', 'TurnRightLarge');

% Додавання вихідної змінної для кута повороту крану холодної води
fuzzy_system = addOutput(fuzzy_system, [-90 90], 'Name', 'ColdWaterValve');
fuzzy_system = addMF(fuzzy_system, 'ColdWaterValve', 'trapmf', [-90 -90 -45 0], 'Name', 'TurnLeftMedium');
fuzzy_system = addMF(fuzzy_system, 'ColdWaterValve', 'trimf', [-45 0 45], 'Name', 'TurnRightMedium');
fuzzy_system = addMF(fuzzy_system, 'ColdWaterValve', 'trapmf', [0 45 90 90], 'Name', 'TurnRightLarge');

% Визначення правил на основі умов задачі
ruleList = [ ...
    3 3 2 2 1 1; % Якщо вода гаряча і її напір сильний (правило 1)
    3 2 1 2 1 1; % Якщо вода гаряча і її напір не дуже сильний (правило 2)
    2 3 2 1 1 1; % Якщо вода не дуже гаряча і її напір сильний (правило 3)
    2 1 1 1 1 1; % Якщо вода не дуже гаряча і її напір слабкий (правило 4)
    2 2 0 0 1 1; % Якщо вода тепла і її напір не дуже сильний (правило 5)
    1 3 2 1 1 1; % Якщо вода прохолодна і її напір сильний (правило 6)
    1 2 2 1 1 1; % Якщо вода прохолодна і її напір не дуже сильний (правило 7)
    1 1 2 0 1 1; % Якщо вода холодна і її напір слабкий (правило 8)
    1 3 2 2 1 1; % Якщо вода холодна і її напір сильний (правило 9)
    2 3 1 1 1 1; % Якщо вода тепла і її напір сильний (правило 10)
    2 1 1 1 1 1; % Якщо вода тепла і її напір слабкий (правило 11)
];

fuzzy_system = addRule(fuzzy_system, ruleList);

% Відображення нечіткої системи для перевірки
showrule(fuzzy_system)
