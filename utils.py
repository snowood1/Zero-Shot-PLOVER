import numpy as np
import itertools
import torch
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import classification_report, confusion_matrix

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


LABEL_DICT = pd.read_csv('./prompts/rootcode_modality.txt', header=0, delimiter='\t')
RootToPenta = dict(zip(LABEL_DICT.rootcode, LABEL_DICT.pentacode))

def decide_root_penta(rootcode, tense, LABEL_DICT=LABEL_DICT):
    pred_root = LABEL_DICT.loc[LABEL_DICT.rootcode == rootcode, tense].item()
    pred_penta = RootToPenta[pred_root]
    return pred_penta, pred_root

penta_to_binary = np.vectorize(lambda t: 0 if t < 3 else 1)

def replace_mask(s, source=None, target=None):
    try:
        s = s.replace('<S>', source)
        s = s.replace('<T>', target)
        s = s.replace('  ', ' ').strip()
        s = s[0].upper() + s[1:] + '.'
        return s
    except:
        return s


def get_nli_scores(premise, hypothesis, tokenizer, nli_model):
    if type(premise) == str:
        premise = [premise] * len(hypothesis)

    encoding = tokenizer(premise, hypothesis, return_tensors='pt', max_length=128,
                         truncation='only_first', padding=True)
    input_ids = encoding['input_ids'].cuda()
    attention_mask = encoding['attention_mask'].cuda()
    with torch.no_grad():
        output = nli_model(input_ids=input_ids, attention_mask=attention_mask)
    #     entail_contradiction_logits = output.logits   #  TODO: transformers version !
    entail_contradiction_logits = output[0]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:, 2]
    prob_label_is_true = prob_label_is_true.cpu()
    return prob_label_is_true.numpy()

def run_all_nli_prompt(sentence, s, t, df_prompt_flatten, tokenizer, nli_model):
    df = df_prompt_flatten.copy()
    df.prompt_text = df.prompt_text.apply(lambda x: replace_mask(x, s, t)).values
    hypothesis = df.prompt_text.tolist()
    nli_scores = get_nli_scores(sentence, hypothesis, tokenizer, nli_model)
    return nli_scores

def load_nli_model(model_name='roberta-large-mnli'):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nli_model.cuda()
    nli_model.eval()
    print("\nNLI Model Loaded")
    return tokenizer, nli_model


## For Debug

# from utils import *
# online=True
# saved_score=None
# consult_penalty=0.02
# cut_threhold=0.1
# log=True
# peace_overwrite=True
# blockade_overwrite=True
# conflict_overwrite=True
# expel_overwrite=True
# Verbal_Conflict=["REQUEST", "ACCUSE"]
# Material_Conflict=["ASSAULT", "PROTEST", "SANCTION", "COERCE"]

def inference(processed_data,
              apply_level2,
              tense_L1,
              tense_L2,
              df_prompt_flatten,
              online,
              saved_score,
              tokenizer,
              nli_model,
              consult_penalty=0.02,
              cut_threhold=0.1,
              log=False,
              peace_overwrite=True,
              blockade_overwrite=True,
              conflict_overwrite=True,
              expel_overwrite=True,
              Verbal_Conflict=["REQUEST", "ACCUSE"],
              Material_Conflict=["ASSAULT", "PROTEST", "SANCTION", "COERCE"]
              ):

    assert online != (type(saved_score) == np.ndarray)

    penta_L1_all = []
    penta_L2_all = []
    root_L1_all = []
    root_L2_all = []
    prompt_L1_all = []
    prompt_L2_all = []
    modality_L2_all = []

    for index in processed_data.index:
        if log:
            print(
                f'\n===================== index: {index} ==============================================================')
            print(processed_data.loc[index, ["sentence", "source", "target"]].to_string(header=False))

        sentence = processed_data.loc[index, "sentence"]
        gold_penta = int(processed_data.at[index, "gold_penta"]) if "gold_penta" in processed_data else None
        gold_root = processed_data.at[index, "gold_root"] if "gold_root" in processed_data else None

        s = processed_data.loc[index, "source"]
        t = processed_data.loc[index, "target"]

        df_flatten = df_prompt_flatten.copy()
        df_flatten['sentence'] = df_flatten['prompt_text'].apply(lambda x: replace_mask(x, s, t))

        if online:
            df_flatten['score'] = None
            hypothesis = df_flatten.loc[df_flatten.tense == tense_L1, 'sentence'].to_list()
            df_flatten.loc[df_flatten.tense == tense_L1, 'score'] = \
                get_nli_scores(sentence, hypothesis, tokenizer, nli_model)
        else:
            df_flatten['score'] = saved_score[index]

        # penalty
        df_flatten.loc[(df_flatten.rootcode == "CONSULT") & (df_flatten.tense == tense_L1),
                       "score"] -= consult_penalty

        df_L1 = df_flatten.loc[(df_flatten.tense == tense_L1)]
        df_L1 = df_L1.sort_values(by="score", ascending=False)
        max_score = df_L1.score.max()
        df_L1 = df_L1[df_L1.score > max_score - cut_threhold].head(5)

        root_L1 = df_L1.rootcode.values[0]
        penta_L1 = RootToPenta[root_L1]
        prompt_L1 = df_L1.prompt_text.tolist()

        root_L1_all.append(root_L1)
        penta_L1_all.append(penta_L1)
        prompt_L1_all.append(prompt_L1)

        if log:
            print('\n-----------------------------level 1 ----------------------------------------------------------')
            print(df_L1[['tense', 'prompt_text', 'rootcode', 'score']].to_string())
            if gold_root:
                print(f"\n Level 1 root gold : {gold_root} \tvs. pred: {root_L1} \t==1c=> {gold_root == root_L1}")
            if gold_penta:
                print(f" Level 1 penta gold : {gold_penta} \t\tvs. pred: {penta_L1} \t\t==1p=> {gold_penta == penta_L1}")

        if not apply_level2:
            continue

        # ==================== Class disambiguation  ==================== #

        if peace_overwrite:
            if len(df_L1[df_L1.prompt_text.str.contains("peace")]) > 0:
                df_L1 = df_L1[~(df_L1.prompt_text.str.contains("forces") & ~df_L1.prompt_text.str.contains("peace"))]
                if log:
                    print("\n ========> PEACEKEEPING OVERWRITE")
                    print(df_L1.loc[:, ['prompt_text', 'rootcode', 'score']].to_string())

        if blockade_overwrite:
            if len(df_L1[df_L1.prompt_text.str.contains("blockades")]) * len(
                    df_L1[df_L1.prompt_text.str.contains("protest")]) > 0:
                df_L1 = df_L1[~df_L1.prompt_text.str.contains("blockades")]
                if log:
                    print("\n ========> blockades OVERWRITE")
                    print(df_L1.loc[:, ['prompt_text', 'rootcode', 'score']].to_string())

        if conflict_overwrite:
            if len(df_L1[df_L1.rootcode.isin(Material_Conflict)]) * len(
                    df_L1[df_L1.rootcode.isin(Verbal_Conflict)]) > 0:
                df_L1 = df_L1[~df_L1.rootcode.isin(Verbal_Conflict)]
                if log:
                    print("\n ========> Conflict OVERWRITE")
                    print(df_L1.loc[:, ['prompt_text', 'rootcode', 'score']].to_string())

        if expel_overwrite:
            if len(df_L1[df_L1.prompt_text.str.contains("expel")]) * len(
                    df_L1[df_L1.prompt_text.str.contains("deport")]) > 0:
                df_L1 = df_L1[~df_L1.prompt_text.str.contains("deport")]
                if log:
                    print("\n ========> Expel OVERWRITE")
                    print(df_L1.loc[:, ['prompt_text', 'rootcode', 'score']].to_string())
        # ============================================================ #


        # ==================== Level2 Modality  ==================== #
        select_prompt_idx = df_L1.prompt_idx
        df_L2 = df_flatten[df_flatten.prompt_idx.isin(select_prompt_idx)].copy()


        diff_tense_L2_L1 = tense_L2.copy()
        diff_tense_L2_L1.remove(tense_L1)  # new tense in Level-2 other than Level-1's
        L2_index = df_L2.loc[df_L2.tense.isin(diff_tense_L2_L1)].index

        if len(L2_index):
            if online:
                hypothesis = df_L2.loc[L2_index, 'sentence'].to_list()
                df_L2.loc[L2_index, 'score'] = \
                    get_nli_scores(sentence, hypothesis, tokenizer, nli_model)
            # penalty
            df_L2.loc[(df_L2.rootcode == "CONSULT") & (df_L2.tense.isin(diff_tense_L2_L1)),
                      "score"] -= consult_penalty

        df_L2 = df_L2.pivot(index=['rootcode','prompt_idx'],
                            columns=['tense'], values='score') \
                            .fillna(-1).reset_index() \
                            .rename_axis(None, axis=1)
        df_L2['prompt_text'] = \
            df_prompt_flatten.loc[(df_prompt_flatten.tense == tense_L1) & \
                df_prompt_flatten.prompt_idx.isin(df_L2.prompt_idx),
                                  'prompt_text'].values

        # In case there are only a subset of L2 tense
        valid_tense_L2 = list(set(tense_L2) & set(df_L2.columns))
        df_L2["l2_mod"] = df_L2[valid_tense_L2].idxmax(axis=1)
        df_L2["l2_score"] = df_L2[valid_tense_L2].max(axis=1)

        df_L2[['l2_penta', 'l2_root']] = \
            df_L2.apply(lambda x: decide_root_penta(x.rootcode, x.l2_mod), axis=1).to_list()

        df_L2 = df_L2.sort_values(by='l2_score', ascending=False)
        root_L2 = df_L2.l2_root.values[0]
        penta_L2 = df_L2.l2_penta.values[0]
        prompt_L2 = df_L2.prompt_text.tolist()
        modality_L2 = df_L2.l2_mod.values[0]

        if log:
            print('\n-----------------------------level 2 ----------------------------------------------------------')
            print(df_L2[['rootcode','prompt_text'] + valid_tense_L2 + ['l2_root']].to_string())
            if gold_root:
                print(f"\n Level 2 root gold : {gold_root} \tvs. pred: {root_L2} \t==2c=> {gold_root == root_L2}")
            if gold_penta:
                print(f" Level 2 penta gold : {gold_penta} \t\tvs. pred: {penta_L2} \t\t==2p=> {gold_penta == penta_L2}")

        root_L2_all.append(root_L2)
        penta_L2_all.append(penta_L2)
        prompt_L2_all.append(prompt_L2)
        modality_L2_all.append(modality_L2)

    out_df = processed_data.copy()

    out_df['binary_L1'] = penta_to_binary(penta_L1_all)
    out_df['penta_L1'] = penta_L1_all
    out_df['root_L1'] = root_L1_all
    out_df['prompt_L1'] = prompt_L1_all

    if apply_level2:
        out_df['binary_L2'] = penta_to_binary(penta_L2_all)
        out_df['penta_L2'] = penta_L2_all
        out_df['root_L2'] = root_L2_all
        out_df['prompt_L2'] = prompt_L2_all
        out_df['modality_L2'] = modality_L2_all

    return out_df


def print_result(out_df, level_idx="L1", log=False):
    result = {}

    binary_L1_all = out_df[f"binary_{level_idx}"].values
    penta_L1_all = out_df[f"penta_{level_idx}"].values
    root_L1_all = out_df[f"root_{level_idx}"].values

    # Binary

    if ("gold_binary" not in out_df) & ('gold_penta' in out_df):
        out_df['gold_binary'] = penta_to_binary(out_df.gold_penta)

    if "gold_binary" in out_df:
        binary_golds = out_df.gold_binary.values
        target_names = ['Coop.', 'Confl.']
        result[f'binary_{level_idx}'] = classification_report(binary_golds, binary_L1_all, target_names=target_names,
                                                    output_dict=True)
        if log:
            print(f"\n{level_idx} report: ")
            print(classification_report(binary_golds, binary_L1_all, target_names=target_names, digits=3))
            print(f"\n{level_idx} Confusin Matrix: ")
            cm = confusion_matrix(binary_golds, binary_L1_all)
            print(cm)

            plt.figure(figsize=(2, 3));
            plt.imshow(cm, interpolation='nearest', norm=matplotlib.colors.PowerNorm(gamma=1 / 3));
            plt.title(f"Binary {level_idx}")

            thresh = cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="red" if cm[i, j] > thresh else "yellow")

            tick_marks = np.arange(len(target_names));
            plt.xticks(tick_marks, target_names, rotation=0);
            plt.yticks(tick_marks, target_names);
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()

    # Pentacode
    if "gold_penta" in out_df:
        penta_golds = out_df.gold_penta.values
        target_names = ['V-Coop.', 'M-Coop.', 'V-Confl.', 'M-Confl.']
        result[f'penta_{level_idx}'] = classification_report(penta_golds, penta_L1_all, target_names=target_names, output_dict=True)

        if log:
            print(f"\n{level_idx} report: ")
            print(classification_report(penta_golds, penta_L1_all, target_names=target_names, digits=3))
            print(f"\n{level_idx} Confusin Matrix: ")
            cm = confusion_matrix(penta_golds, penta_L1_all)
            print(cm)

            plt.figure(figsize=(3.5, 4));
            plt.imshow(cm, interpolation='nearest', norm=matplotlib.colors.PowerNorm(gamma=1 / 3));
            plt.title(f"Penta {level_idx}")

            thresh = cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="red" if cm[i, j] > thresh else "yellow")

            tick_marks = np.arange(len(target_names));
            plt.xticks(tick_marks, target_names, rotation=0);
            plt.yticks(tick_marks, target_names);
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()

    # Rootcode
    if "gold_root" in out_df:
        root_golds = out_df.gold_root.values
        result[f'root_{level_idx}'] = classification_report(root_golds, root_L1_all, output_dict=True)

        target_names = ["AGREE",
                        "CONSULT",
                        "SUPPORT",
                        "COOPERATE",
                        "AID",
                        "YIELD",
                        "ACCUSE",
                        "REQUEST",
                        "REJECT",
                        "THREATEN",
                        "PROTEST",
                        "MOBILIZE",
                        "SANCTION",
                        "COERCE",
                        "ASSAULT"]

        if log:
            print(f"\n{level_idx} report: ")
            print(classification_report(root_golds, root_L1_all, digits=3))
            print(f"\n{level_idx} Confusin Matrix: ")
            cm = confusion_matrix(root_golds, root_L1_all, labels=target_names)
            print(cm)

            plt.figure(figsize=(8, 7));
            plt.imshow(cm, interpolation='nearest', norm=matplotlib.colors.PowerNorm(gamma=1 / 3));
            plt.title(f"root {level_idx}")
            plt.colorbar(shrink=0.8);

            thresh = cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="red" if cm[i, j] > thresh else "yellow")

            tick_marks = np.arange(len(target_names));
            plt.xticks(tick_marks, target_names, rotation=45);
            plt.yticks(tick_marks, target_names);
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
    return result

