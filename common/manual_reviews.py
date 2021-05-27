import pandas as pd

"""
Script containing manually written reviews. 
"""

data = {'Hotel_Address': ['Nieuwendijk 17, 4381 BV Vlissingen, Nederland',
                          'Nieuwendijk 17, 4381 BV Vlissingen, Nederland',
                          'Nieuwendijk 17, 4381 BV Vlissingen, Nederland'],
        'Additional_Number_of_Scoring': ['1523', '1523', '1523'],
        'Review_Date': ['2/6/2020', '4/23/2020', '7/23/2020'],
        'Average_Score': [7.8, 8.8, 8.8],
        'Hotel_Name': ['Lotte Hotel Seoul', 'Lotte Hotel Seoul', 'Lotte Hotel Seoul'],
        'Reviewer_Nationality': ['United Kingdom', 'China', 'Australia'],
        'Negative_Review': ['Very bad cleaning.', 'Strange smell in living room.', 'Very small space.'],
        'Review_Total_Negative_Word_Counts': [3, 5, 3],
        'Total_Number_of_Reviews': [420, 420, 420],
        'Positive_Review': ['Really good service.', 'I had a really comfortable experience.',
                            'The location was a sight to behold'],
        'Review_Total_Positive_Word_Counts': [3, 6, 4],
        'Total_Number_of_Reviews_Reviewer_Has_Given': [4, 14, 7],
        'Reviewer_Score': [8, 8.5, 9],
        'Tags': 'no tags',
        'days_since_review': [227, 150, 59],
        'lat': [51.441470, 51.441470, 51.441470],
        'lng': [3.575640, 3.575640, 3.575640]
        }

own_review_df = pd.DataFrame(data,
                             columns=['Hotel_Address', 'Additional_Number_of_Scoring', 'Review_Date',
                                      'Average_Score', 'Hotel_Name', 'Reviewer_Nationality',
                                      'Negative_Review', 'Review_Total_Negative_Word_Counts',
                                      'Total_Number_of_Reviews', 'Positive_Review',
                                      'Review_Total_Positive_Word_Counts',
                                      'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score',
                                      'Tags', 'days_since_review',
                                      'lat', 'lng'])
own_review_df.to_csv('Hotel_Review_Manual.csv', index=False)
