Command options

base {path} - Trains a base model from jellybeans, saves to path

{network|control} {path} {folder} - Creates a network in the {network|control} condition of slightly different models based on base model {path}, stored in folder {folder}

eval {folder} {res_json} - evaluates network in {folder}, stores results in {res_json}

rn {folder} {img} {res} - reinforces network in {folder} based on {img} for the experimental condition, using {res} as an intermediate .json

rc {folder} {img} {res} - reinforces network in {folder} based on {img} for the control condition, using {res} as an intermediate .json
