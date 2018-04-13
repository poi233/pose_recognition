import json
import utils
import os
import process_and_save

# get json files to result_json
result_json = {}
json_path = os.path.join('result', '{}/json'.format('example_1_result'))
json_file = os.listdir(json_path)[0]
with open(os.path.join(json_path, json_file), 'r') as json_file:
    json_data = json.load(json_file)
    for human_id, data in json_data.items():
        avg_vertical_leg = utils.get_average(data['theta_vertical_8_9'], data['theta_vertical_11_12'])
        avg_leg = utils.get_average(data['theta_1_8_9'], data['theta_1_11_12'])
        avg_knee = utils.get_average(data['theta_8_9_10'], data['theta_11_12_13'])
        avg = utils.get_average(utils.get_average(avg_vertical_leg, avg_leg), avg_knee)
        json_data[human_id]['avg_vertical_leg'] = avg_vertical_leg
        json_data[human_id]['avg_leg'] = avg_leg
        json_data[human_id]['avg_knee'] = avg_knee
        json_data[human_id]['avg'] = avg

process_and_save.save_action_time('example_1', json_data, smooth_rate=0.1, threshold=0.1)