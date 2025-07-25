import os
import pandas as pd
from io import StringIO
import time
from pydantic import BaseModel
from google import genai
from google.genai import types
import contextlib
import random
import json

#Data Format
class MagneticMaterial(BaseModel):
  chemical_composition: str
  crystal_structure: str | None
  centering: str | None
  lattice_parameters: str | None
  space_group: str | None
  curie_temperature: str | None
  neel_temperature: str | None
  magnetic_moment: str | None
  magnetocrystalline_anisotropy: str | None
  experimental: bool | None

def extract_magnetic_properties(text, properties_and_explanations, example_entry, output_structure):
    #Lab key
    #google_api_key = 

    #Ian's key
    google_api_key = 'AIzaSyCrdi6xQKZLxFsIhcb9-a6_cLIGIzNRMLU'
    client = genai.Client(api_key=google_api_key)


    property_explanation_string = '\n'.join([key + ' - ' + value for key, value in properties_and_explanations.items()])

    example_output = '\n'.join([','.join(properties_and_explanations.keys()), ','.join(example_entry)])

    system_instructions = f"You are an expert at structured data extraction. You will be given unstructured text from a research paper on magnetic materials and should convert it into the given structure. You leave properties that aren't mentioned for a material blank. You are very careful to be sure of what information relates to which material. You know the following descriptions of the properties you want:\n\n {property_explanation_string}\n\n You also know the following example output: \n\n {example_output} \n\n"
    content = f"<paper_start>\n\n{text}\n\n<paper_end>"

    response = client.models.generate_content(
    model="gemini-2.5-flash-lite", # can we use 2.5? Yup
    config= types.GenerateContentConfig(system_instruction=system_instructions, response_mime_type='application/json', response_schema=output_structure),
    contents=content
    )
    return(response.text)

#List of properties and their explanations, which will be passed to the model for improved extraction
properties_and_explanations = {'chemical_composition':'Represents what elements are in the material and their ratios', 
                            'crystal_structure':'The distinctive arrangement of atoms, molecules, or ions in a crystal. It is highly ordered and repetitive, creating a characteristic pattern that defines the crystal’s shape and properties.', 
                            'centering':'Describes the placement of atoms in the unit cell, particularly in terms of what part of the cell they are centered around.',
                            'lattice_parameters':'Represent the lengths of the lattice vectors, which define the unit cell. There can be up to three lattice vectors for a unit cell.',
                            'space_group':'Describes the possible symmetry operations of the material based on its structure. An example space group is - I4/mmm',
                            'curie_temperature':'Is the temperature below which a magnetic material becomes ferromagnetic.',
                            'neel_temperature':'Is the temperature below which a magnetic material becomes anti-ferromagnetic.',
                            'magnetic_moment':'Describes the strength and direction of a magnet.',
                            'magnetocrystalline_anisotropy':'Desribes the the directional dependence of a material\'s magnetic properties due to the crystal structure.',
                            'experimental':'True if the values were measured experimentally, false otherwise.'}

#Example entry to help with format of output values (not currently used)
example_entry = ['CeMgPb', 'Tetragonal', 'Face Centered', 'a:4.557; c:16.405', 'I4/mmm', '7 k', '30 k', '2.25 J/T', '48000 J/m^3', 'True']

#The dataframe which we will be populating during extraction
df = pd.DataFrame(columns=list(properties_and_explanations.keys()) + ['ID'])

#Maximum number of papers from the source directory to run extraction on
max_num_papers = 4350

#Source directory holding the text forms of papers to extract
source_directory = 'C:/Users/haoze/Documents/Database_Generation/MD_Files/JMMM'

#Generates a list of all files to be extracted
dir_files = os.listdir(source_directory)

#Path to output directory
output_dir = 'C:/Users/haoze/Documents/Database_Generation/DataBases/JMMM/Initial_Tests'

#Name of output csv file
output_file = f'JMMM_Initial_Test_example_entry_{min(max_num_papers, len(dir_files))}'

#Number of requests allowed per minute
max_requests_per_minute = 15

#Tracks timestamps for prior extractions (this is necessary because some api's restrict the number of calls per minute)
times = [-60 for i in range(max_requests_per_minute)]

#Track the start time so we can measure the length of the entire run
start_time = time.time()

#Loop through each paper, extracting data and adding it to the dataframe
for file in dir_files[:max_num_papers]:
    #The text of the research article we will be extracting from
    text = open(f'{source_directory}/{file}', encoding="utf8").read()
    print(file)

    try:
      #Run extraction and recieve output as a string in json format
      output = extract_magnetic_properties(text, properties_and_explanations, example_entry, list[MagneticMaterial])

      #Convert the json format string to a dataframe
      output_df = pd.DataFrame(json.loads(output))
    
    except Exception as e:
      #Record the file for which extraction failed, as well as the error
      with open(f'{output_dir}/{output_file}_failed_attempts', 'a') as error_file:
         error_file.write(f'File Name: {file}\nError: {e}\n\n')
      
      #Update the list of times, and wait till the oldest in the list is greater than 60 seconds
      times.pop(0)
      times.append(time.time())
      while time.time() - times[0] < 61:
        time.sleep(1)
      
      continue

    #If the output carries new entries, add them to the database
    if len(output_df) > 0:
      output_df['ID'] = file
      print(output_df)
      df = pd.concat([df, output_df])
      df.to_csv(f'{output_dir}/{output_file}_in_progress.csv')
    
    #Update the list of times, and wait till the oldest in the list is greater than 60 seconds
    times.pop(0)
    times.append(time.time())
    while time.time() - times[0] < 61:
      time.sleep(1)
    

    print('-------------')

df.to_csv(f'{output_dir}/{output_file}_final.csv')