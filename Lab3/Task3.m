% Створення нечіткої системи
fuzzy_system = mamfis('Name', 'AirConditionerControl');

% Додавання вхідної змінної для температури
fuzzy_system = addInput(fuzzy_system, [0 40], 'Name', 'Temperature');
fuzzy_system = addMF(fuzzy_system, 'Temperature', 'trapmf', [0 0 10 15], 'Name', 'VeryCold');
fuzzy_system = addMF(fuzzy_system, 'Temperature', 'trimf', [10 15 20], 'Name', 'Cold');
fuzzy_system = addMF(fuzzy_system, 'Temperature', 'trimf', [15 20 25], 'Name', 'Normal');
fuzzy_system = addMF(fuzzy_system, 'Temperature', 'trimf', [20 25 30], 'Name', 'Warm');
fuzzy_system = addMF(fuzzy_system, 'Temperature', 'trapmf', [25 30 40 40], 'Name', 'VeryWarm');

% Додавання вхідної змінної для швидкості зміни температури
fuzzy_system = addInput(fuzzy_system, [-5 5], 'Name', 'TemperatureChange');
fuzzy_system = addMF(fuzzy_system, 'TemperatureChange', 'trapmf', [-5 -5 -2 0], 'Name', 'Negative');
fuzzy_system = addMF(fuzzy_system, 'TemperatureChange', 'trimf', [-2 0 2], 'Name', 'Zero');
fuzzy_system = addMF(fuzzy_system, 'TemperatureChange', 'trapmf', [0 2 5 5], 'Name', 'Positive');

% Додавання вихідної змінної для керування кондиціонером
fuzzy_system = addOutput(fuzzy_system, [-90 90], 'Name', 'ACControl');
fuzzy_system = addMF(fuzzy_system, 'ACControl', 'trapmf', [-90 -90 -45 0], 'Name', 'CoolStrongLeft');
fuzzy_system = addMF(fuzzy_system, 'ACControl', 'trimf', [-45 0 45], 'Name', 'CoolWeakLeft');
fuzzy_system = addMF(fuzzy_system, 'ACControl', 'trimf', [0 45 90], 'Name', 'HeatWeakRight');
fuzzy_system = addMF(fuzzy_system, 'ACControl', 'trapmf', [45 90 90 90], 'Name', 'HeatStrongRight');
fuzzy_system = addMF(fuzzy_system, 'ACControl', 'trimf', [-45 0 45], 'Name', 'Off');

% Визначення правил на основі умов задачі
ruleList = [ ...
    5 3 1 1 1; % Якщо дуже тепла температура і швидкість зміни додатня
    5 1 2 1 1; % Якщо дуже тепла температура і швидкість зміни від'ємна
    4 3 1 1 1; % Якщо тепла температура і швидкість зміни додатня
    4 1 5 1 1; % Якщо тепла температура і швидкість зміни від'ємна
    1 1 4 1 1; % Якщо дуже холодна температура і швидкість зміни від'ємна
    1 3 3 1 1; % Якщо дуже холодна температура і швидкість зміни додатня
    2 1 4 1 1; % Якщо холодна температура і швидкість зміни від'ємна
    2 3 5 1 1; % Якщо холодна температура і швидкість зміни додатня
    5 2 1 1 1; % Якщо дуже тепла температура і швидкість зміни 0
    4 2 2 1 1; % Якщо тепла температура і швидкість зміни 0
    1 2 4 1 1; % Якщо дуже холодна температура і швидкість зміни 0
    2 2 3 1 1; % Якщо холодна температура і швидкість зміни 0
    3 3 2 1 1; % Якщо нормальна температура і швидкість зміни додатня
    3 1 3 1 1; % Якщо нормальна температура і швидкість зміни від'ємна
    3 2 5 1 1; % Якщо нормальна температура і швидкість зміни 0
];

fuzzy_system = addRule(fuzzy_system, ruleList);

% Відображення правил для перевірки
showrule(fuzzy_system)
