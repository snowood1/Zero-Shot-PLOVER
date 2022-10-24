import numpy as np
import pandas as pd
# pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 0)
import warnings
warnings.simplefilter("ignore", UserWarning)
import argparse
from utils import inference, print_result, load_nli_model, run_all_nli_prompt


# ======== For debug  ======== #

# data_dir = args.dataset
# prompt_dir = args.prompt_dir
# score_dir = args.score_dir
# model_name = args.model_name
# output_dir = args.output_dir
# consult_penalty = args.consult_penalty
# infer_setting = args.infer_setting
# run_offline_nli = args.run_offline_nli
# write_score_result = args.write_score_result
#

# data_dir = './datasets/PLV_test.tsv'        # PLV_test,  AW_test
# prompt_dir = "./prompts/Tree.txt"          # Tiny,  Full,  Tree
# RESULT_DIR = "./results/PLV_test-Tree.npy"
#
# model_name = "roberta-large-mnli"
# consult_penalty = 0.02

# infer_setting = "offline"    # offline, online
# run_offline_nli = True
# write_offline_result = False

# infer_setting = "online"    # offline, online


def main(
        data_dir="./datasets/PLV_test.tsv",
        prompt_dir="./prompts/Tree.txt" ,
        score_dir="./scores/PLV_test-Tree.npy",
        model_name="roberta-large-mnli",
        output_dir="./outputs/PLV_test-Tree-result.csv",
        consult_penalty=0.02,
        infer_setting="offline",
        run_offline_nli=False,
        write_score_result=False,
        log=True
):

    print("\n ==== Inference setting:  ==== ")
    if infer_setting == "offline":
        print(" > offline inference")
        if run_offline_nli == True:
            print(" >> run nli scores from scratch")
            print(f" >> save nli scores?  {write_score_result}   {score_dir if write_score_result else ''}")
        else:
            print(f" >> load saved nli scores from {score_dir}")
    else:
        print(" > online inference")

    # ======== Data ======== #
    if 'PLV' in data_dir:
        processed_data = pd.read_csv(data_dir, delimiter='\t')

    if 'AW' in data_dir:
        processed_data = pd.read_csv(data_dir, delimiter='\t')
        processed_data.source = processed_data.source.apply(lambda x: x.split(":")[-1])
        processed_data.target = processed_data.target.apply(lambda x: x.split(":")[-1])
        processed_data = processed_data[['gold_binary','gold_penta', 'gold_root', 'event_type', 'sentence', 'source', 'target']]

    # ======== Prompts ======== #
    df_prompt = pd.read_csv(prompt_dir, header=0, delimiter='\t')
    df_prompt.pentacode = df_prompt.pentacode.apply(lambda x: [int(i) for i in x.split(',')] if type(x) == str else x)

    # merge some CAMEO Rootcode to PLOVER Rootcode
    df_prompt.loc[df_prompt.rootcode.isin(['INVESTIGATE']), "rootcode"] = "ACCUSE"
    df_prompt.loc[df_prompt.rootcode.isin(['FIGHT']), "rootcode"] = "ASSAULT"
    print("\nPrompt Rootcode unique:")
    print(set(df_prompt.rootcode.unique()))

    # ======== Modality ======== #
    TENSE = df_prompt.columns[2:].to_list()
    tense_L1 = 'past'
    tense_L2 = TENSE.copy()

    print("\nTENSE:\t\t", TENSE)
    print("tense_L1:\t", tense_L1)
    print("tense_L2:\t", tense_L2)

    # ======== all hypothesis ======== #
    prompt_text = df_prompt[TENSE] \
        .stack().reset_index().rename(columns={0: 'prompt_text', 'level_0': 'prompt_idx', 'level_1': 'tense'})
    root_flatten = df_prompt.rootcode.repeat(len(TENSE)).reset_index(drop=True)
    penta_flatten = df_prompt.pentacode.repeat(len(TENSE)).reset_index(drop=True)
    df_prompt_flatten = pd.concat([root_flatten, penta_flatten, prompt_text], axis=1)
    df_prompt_flatten = df_prompt_flatten[df_prompt_flatten.prompt_text != "None"].reset_index(drop=True)
    print(f"\nall prompts flatten:\n{df_prompt_flatten.iloc[np.r_[0:4, -4:0]].to_string()}")

    if infer_setting == "offline":
        print('\nStart offline inference...')
        if run_offline_nli:
            # ======== model ======== #
            print("\nLoading models...")
            tokenizer, nli_model = load_nli_model(model_name)

            # ======== Run ======== #
            print("\nRun inference on all prompts...")

            result = []
            for index in range(len(processed_data)):
                print(f"{index+1}/{len(processed_data)}", end='\r')
                sentence = processed_data.loc[index, "sentence"]
                s = processed_data.loc[index, "source"]
                t = processed_data.loc[index, "target"]
                result.append(run_all_nli_prompt(sentence, s, t, df_prompt_flatten, tokenizer, nli_model))
            result = np.stack(result, axis=0)

            if write_score_result:
                print(f"\nSave scores for offline analysis at {score_dir}")
                with open(score_dir, 'wb') as f:
                    np.save(f, result)

            # ======== Offline inference ======== #
            saved_score = result
            print(f"\nOffline inference from the NLI scores we just got...")

        else:
            # ======== Offline inference ======== #
            saved_score = np.load(score_dir)
            print(f"\nOffline inference from the saved scores at {score_dir}")

        out_df = inference(processed_data,
                           TENSE=TENSE,
                           tense_L1=tense_L1,
                           tense_L2=tense_L2,
                           df_prompt_flatten=df_prompt_flatten,
                           online=False,
                           saved_score=saved_score,
                           tokenizer=None,
                           nli_model=None,
                           consult_penalty=consult_penalty,
                           log=log,
                           peace_overwrite=True,
                           blockade_overwrite=True,
                           conflict_overwrite=True,
                           expel_overwrite=True)

    else:
        # ======== model ======== #
        print('\nStart online inference...')
        print("\nLoading models...")
        tokenizer, nli_model = load_nli_model(model_name)

        out_df = inference(processed_data,
                           TENSE=TENSE,
                           tense_L1=tense_L1,
                           tense_L2=tense_L2,
                           df_prompt_flatten=df_prompt_flatten,
                           online=True,
                           saved_score=None,
                           tokenizer=tokenizer,
                           nli_model=nli_model,
                           consult_penalty=consult_penalty,
                           log=log,
                           peace_overwrite=True,
                           blockade_overwrite=True,
                           conflict_overwrite=True,
                           expel_overwrite=True)

    print(f"\n\n ===== Saving to {output_dir} ===== ")
    if output_dir:
        out_df.to_csv(output_dir, index=False)
    print("\n\n ===== Summary ===== \n")
    summary = print_result(out_df, log)

    return out_df, summary


if __name__ == '__main__':
    # ======== Setting ======== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="./datasets/PLV_test.tsv")
    parser.add_argument("--prompt_dir", type=str, default="./prompts/Tree.txt")
    parser.add_argument("--score_dir", type=str, default="./scores/PLV_test-Tree.npy")
    parser.add_argument("--model_name", default="roberta-large-mnli", type=str)
    parser.add_argument("--consult_penalty", default=0.02, type=float)
    parser.add_argument("--infer_setting", type=str, choices=['online', 'offline'])
    parser.add_argument("--run_offline_nli", default=False, type=bool)
    parser.add_argument("--write_score_result", default=False, type=bool)
    parser.add_argument("--output_dir", type=str, default="./outputs/PLV_test-Tree-result.csv")
    parser.add_argument("--log", type=bool, default=True)
    args = parser.parse_args()

    main(data_dir=args.dataset,
         prompt_dir=args.prompt_dir,
         score_dir=args.score_dir,
         model_name=args.model_name,
         output_dir=args.output_dir,
         consult_penalty=args.consult_penalty,
         infer_setting=args.infer_setting,
         run_offline_nli=args.run_offline_nli,
         write_score_result=args.write_score_result,
         log=args.log)

# ======== For debug  ======== #

# data_dir = args.dataset
# prompt_dir = args.prompt_dir
# score_dir = args.score_dir
# model_name = args.model_name
# output_dir = args.output_dir
# consult_penalty = args.consult_penalty
# infer_setting = args.infer_setting
# run_offline_nli = args.run_offline_nli
# write_score_result = args.write_score_result
#

# data_dir = './datasets/PLV_test.tsv'        # PLV_test,  AW_test
# prompt_dir = "./prompts/Tree.txt"          # Tiny,  Full,  Tree
# RESULT_DIR = "./results/PLV_test-Tree.npy"
#
# model_name = "roberta-large-mnli"
# consult_penalty = 0.02

# infer_setting = "offline"    # offline, online
# run_offline_nli = True
# write_offline_result = False

# infer_setting = "online"    # offline, online
