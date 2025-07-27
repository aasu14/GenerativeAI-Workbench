import re
from faker import Faker
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from config.openai_config import SEP

fake = Faker()

'''
exclusion_list = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March",
    "April", "May", "June", "July", "August", "September", "October", "November", "December", "zero", "one", "two",
    "three", "four", "five", "six", "seven", "eight", "nine", "ten", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th",
    "8th", "9th", "10th", "dollar", "dollars", "rupee", "rupees", "hundred", "thousand", "million", "billion",
    "trillion", 'P2P', 'week', 'weeks', 'Today', 'Yesterday', 'Tomorrow', 'DOS', 'DOB', 'UB04', '-', "01", "02", "03",
    "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22",
    "23", "24", "25", "26", "27", "28", "29", "30", "31"]

# Add years, cpt, diag, proc and rev code to the exclusion list
exclusion_list += [str(year) for year in range(1950, 2041)]

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
rev_file = os.path.join(CUR_PATH, '../../..', 'data', 'RevGrouping.csv')
exclusion_list += pd.read_csv(f'{rev_file}', sep='|', header=None)[0].tolist()[1:]
exclusion_list = list(map(lambda x: x.lower(), exclusion_list))
'''
exclusion_list = []


class PromptCleanser:
    def __init__(self, remit_cas_code_list=[]):
        # Load the spaCy language model
        self.preserve_tokens = None
        # self.nlp = spacy.load("en_core_web_trf")
        # self.nlp.max_length = 5000000
        # self.remit_cas_code_list = remit_cas_code_list

    def _anonymize_token(self, doc):
        anonymized_text, skip_name, skip_gpe, skip_date = [], False, False, False
        anonym_dict = {}

        for i, token in enumerate(doc):
            if (re.match(r"\d{2,4}[-/.]\d{2}[-/.]\d{2,4}", token.text) or token.ent_type_ == "DATE" or
            token.ent_type_ == 'TIME') and (str(token.text).lower() not in self.preserve_tokens):
                anonymized_text.append(token.text)
                # Note: Date Anonymized logic
                # fake_date = str(fake.date_between(datetime.strptime('2022/01/01', '%Y/%m/%d'),
                #                                   datetime.strptime('2024/01/01', '%Y/%m/%d')))
                # anonymized_text.append(fake_date) if not skip_date else None
                # if not skip_date:
                #     anonym_dict[fake_date] = self.get_continous_string(doc, i, 'DATETIME') \
                #         if token.ent_type_ in ['DATE', 'TIME'] else token.text
                # skip_date = True
            elif token.ent_type_ == "MONEY":
                anonymized_text.append(token.text)
            elif token.ent_type_ == "PERSON":
                fake_person = fake.name()
                anonymized_text.append(fake_person) if not skip_name else None
                if not skip_name:
                    anonym_dict[fake_person] = self.get_continous_string(doc, i, token.ent_type_)
                skip_name = True
            elif token.ent_type_ == "GPE":
                fake_gpe = fake.country()
                anonymized_text.append(fake_gpe) if not skip_gpe else None
                if not skip_gpe:
                    anonym_dict[fake_gpe] = self.get_continous_string(doc, i, token.ent_type_)
                skip_gpe = True
            elif ((token.like_num or len(re.findall('[0-9]+', f'{token}')) > 0) and
                  (str(token.text).lower() not in self.preserve_tokens)):
                fake_num = str(fake.random_int(min=1000, max=9999999))
                anonymized_text.append(fake_num)
                anonym_dict[fake_num] = token.text
                skip_name, skip_gpe, skip_date = False, False, False
            else:
                anonymized_text.append(token.text)
                skip_name, skip_gpe, skip_date = False, False, False

        return anonymized_text, anonym_dict

    def _token_extraction(self, prompt):
        # Pattern for Amount extraction
        transaction_pattern = r"\$\d{1,3}(?:,?\d{3})*(?:\.\d{1,3})\b"
        transactions = re.findall(transaction_pattern, prompt)
        return [amount.replace('$', '') for amount in transactions] + exclusion_list + self.remit_cas_code_list

    @staticmethod
    def clean_text(text):
        # Remove html tags from notes, symbols, extra whitespace
        soup = BeautifulSoup(text, "html.parser")
        text_content = soup.get_text()
        text = re.sub(r'[#â€™]', '', text_content)
        cleaned_text = re.sub(r'\s+', ' ', text)
        final_text = re.sub(SEP, '\n', cleaned_text)

        return final_text

    @staticmethod
    def get_continous_string(doc, index, entity):
        value = []
        if entity != 'DATETIME':
            for item in doc[index:]:
                if item.ent_type_ == entity:
                    value.append(item.text)
                else:
                    break
        else:
            for item in doc[index:]:
                if (re.match(r"\d{2,4}[-/.]\d{2}[-/.]\d{2,4}", item.text)
                        or item.ent_type_ == "DATE" or item.ent_type_ == 'TIME'):
                    value.append(item.text)
                else:
                    break
        return ' '.join(value)

    def anonymize_text(self, text):
        doc = self.nlp(text)
        self.preserve_tokens = self._token_extraction(text)
        anonymized_text, anonym_dict = self._anonymize_token(doc)
        return " ".join(anonymized_text), anonym_dict

    def prompt_cleansing(self, prompt_text):
        cleaned_text = self.clean_text(prompt_text)
        # anonymized_text, anonym_dict = self.anonymize_text(cleaned_text)
        # anonymized_text = re.sub(r'(\d)\s*-\s*(?=\d)', r'\1-', anonymized_text)
        # return anonymized_text, anonym_dict
        return cleaned_text, {}


if __name__ == "__main__":
    # Test the anonymization
    prompt = f""

    cleanser = PromptCleanser()
    anonymized_prompt = cleanser.prompt_cleansing(prompt)

    print(anonymized_prompt)
