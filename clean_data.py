import pandas as pd
import pdfquery
import matplotlib.pyplot as plt

# transport monthly total boardings for each mode: cleaning etc
total_monthly_24 = pd.read_csv("grand_totals_23-24.csv").drop([0,1,2,3,4,5])
total_monthly_25 = pd.read_csv("grand_totals_24_25.csv").drop([0,1,2,3,4,5])
grand_totals = pd.concat([total_monthly_25, total_monthly_24], ignore_index=True)
grand_totals = grand_totals.rename(columns={"Unnamed: 0":"Months", "Patronage by mode":"Bus", "Unnamed: 2":"Train", "Unnamed: 3":"Ferry","Unnamed: 4":"Grand Total"})
grand_totals = grand_totals.drop(["Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9",
                                  "Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14", "Unnamed: 15",
                                  "Unnamed: 16"], axis=1)
grand_totals = grand_totals.iloc[::-1].reset_index(drop=True)
grand_totals = grand_totals.drop([2])
rows, cols = grand_totals.shape
for i in range(0,rows):
    for j in range(0, cols):
        grand_totals.iat[i,j] = grand_totals.iat[i,j].replace(",", "")
grand_totals[["Bus", "Train", "Ferry", "Grand Total"]] = grand_totals[["Bus", "Train", "Ferry", "Grand Total"]].astype(float)

# September 2023 data dropped due to lack of weather statistics
# monthly rainfall (as measure of weather): cleaning etc


rainfall_monthly = []
rainfall_master = []

pdfs = ["W58EL8-Climate_Statistics_July_2023-NIWA.pdf", "Climate-Statistics-August2023.pdf"
        , "Climate_Statistics_October_2023.pdf", "Climate-Statistics-November-2023-NIWA.pdf", "Climate-Statistics-December-2023-NIWA.pdf",
        "ClimateStatistics-January2024-NIWA.pdf", "Climate-Statistics-February-2024-NIWA.pdf", "ClimateStatistics-March2024.pdf",
        "ClimateStatistics-April2024.pdf", "ClimateStatistics-May2024.pdf", "Climate-Statistics-June 2024-NIWA.pdf",
        "Climate-Statistics-July2024-NIWA.pdf", "Climate Statistics - August 2024.pdf", "Climate-Statistics-September2024-NIWA.pdf",
        "Climate-Statistics-October2024-NIWA.pdf", "ClimateStatistics-November2024-NIWA.pdf", "Climate-Statistics-December-2024-NIWA.pdf",
        "ClimateStatistics-January2025-NIWA.pdf", "Climate Statistics - February 2025.pdf", "Climate-Statistics-March2025-NIWA.pdf",
        "Climate_Statistics-April2025-NIWA.pdf", "Climate Statistics - May 2025.pdf"]

for pdf in pdfs:
    month = pdfquery.PDFQuery(pdf)
    month.load(1)
    month.get_layout(1)
    for k in range(0, 5):
        akl_loc = month.pq('LTTextLineHorizontal:contains("Auckland")')[k]
        bound_box = [float(akl_loc.attrib['x0']), float(akl_loc.attrib['y0']), float(akl_loc.attrib['x1']),
                     float(akl_loc.attrib['y1'])]
        line_els = month.pq('LTTextLineHorizontal')
        relevant_values = [val for val in line_els if abs(float(val.attrib['y0']) - bound_box[1]) < 1]
        for i in range(0, len(relevant_values)):
            for j in range(i + 1, len(relevant_values)):
                if float(relevant_values[j].attrib['x0']) < float(relevant_values[i].attrib['x0']):
                    temp = relevant_values[j]
                    relevant_values[j] = relevant_values[i]
                    relevant_values[i] = temp
        text_list = [val.text for val in relevant_values]
        if len(text_list) > 15:
            num_to_merge = len(text_list) - 15
            text_list = text_list[num_to_merge:]
        if text_list[9] == '. . ':
            pass
        else:
            rainfall_monthly.append(float(text_list[9]))
    rainfall_master.append(sum(rainfall_monthly))
    rainfall_monthly = []
rainfall_df = pd.DataFrame(rainfall_master)
# df of modes + monthly rainfall measures

grand_totals = pd.merge(grand_totals, rainfall_df)

# Exploratory Analysis

grand_totals.plot(title="Rainfall vs Ridership")













