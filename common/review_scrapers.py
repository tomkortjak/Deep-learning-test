from selenium import webdriver
import time
import pandas as pd
import datetime

TODAY = datetime.datetime.now()
DRIVER_PATH = 'assets/chromedriver.exe'


class ReviewScrapers:
    """
    Allows you to scrape with your desired website. Fill the text files in the url directory to scrape your desired
    hotels.
    """
    @staticmethod
    def scrapeBooking():
        """
        Sets up the driver and starts scraping the urls for the required columns. There is also cleaning performed
        on the columns that require it.
        :return: Dataframe containing all the scraped reviews
        """
        # Setting up Chrome driver
        options = webdriver.ChromeOptions()
        options.add_argument("--window-size=800,1200")
        driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH)

        hotel_reviews_scraped = []

        cities_urls = []
        with open('assets/urls/bookinghotelUrls.txt') as webSet:
            for url in webSet:
                cities_urls.append(url.replace('\n', ''))

        for url in cities_urls:
            driver.get(url)

            # Scrape basic info before getting the reviews
            hotel_address = driver.find_element_by_xpath('//*[@id="showMap2"]/span').text
            hotel_name = driver.find_element_by_class_name('hp__hotel-name').text.replace('\n', '')
            total_reviews = driver.find_element_by_class_name('bui-review-score__text').text.split()[0]
            hotel_latlng = driver.find_element_by_xpath('//*[@id="hotel_header"]').get_attribute(
                'data-atlas-latlng').partition(
                ',')
            hotel_lat = hotel_latlng[0]
            hotel_lng = hotel_latlng[2]

            # Simulate click and gather all review elements
            driver.find_element_by_xpath('//*[@id="show_reviews_tab"]').click()

            time.sleep(2)
            average_score = driver.find_element_by_class_name('bui-review-score__badge').text
            reviews = driver.find_elements_by_class_name('c-review-block')

            # Loop over all review elements and gather information
            for review in reviews:
                review_score = review.find_element_by_class_name('bui-review-score__badge').text
                review_nationality = review.find_element_by_class_name('bui-avatar-block__subtitle').text
                reviews_mixed = review.find_elements_by_class_name('c-review__body')
                review_positive = reviews_mixed[0].text.replace('\n', '').encode('ascii', 'ignore').decode('ascii')
                if len(reviews_mixed) == 2:
                    review_negative = reviews_mixed[1].text.replace('\n', '').encode('ascii', 'ignore').decode('ascii')
                else:
                    review_negative = ''
                review_total_positive = len(review_positive.split())
                review_total_negative = len(review_negative.split())
                review_date = review.find_element_by_class_name('c-review-block__date').text.split(': ')[1]
                review_date = pd.to_datetime(review_date, format='%d %B %Y')
                days_since_review = (TODAY - review_date).days

                hotel_reviews_scraped.append(
                    [hotel_address, 10, review_date, average_score, hotel_name, review_nationality, review_negative,
                     review_total_negative, total_reviews, review_positive, review_total_positive,
                     51, review_score, 'no tags', days_since_review, hotel_lat, hotel_lng])

        # Convert the scraped reviews list to a DataFrame
        hotel_reviews_scraped = pd.DataFrame(hotel_reviews_scraped,
                                             columns=['Hotel_Address', 'Additional_Number_of_Scoring', 'Review_Date',
                                                      'Average_Score', 'Hotel_Name', 'Reviewer_Nationality',
                                                      'Negative_Review', 'Review_Total_Negative_Word_Counts',
                                                      'Total_Number_of_Reviews', 'Positive_Review',
                                                      'Review_Total_Positive_Word_Counts',
                                                      'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score',
                                                      'Tags', 'days_since_review',
                                                      'lat', 'lng'])
        driver.quit()
        return hotel_reviews_scraped

    @staticmethod
    def scrapeAgoda():
        """
        Sets up the driver and starts scraping the urls for the required columns. There is also cleaning performed
        on the columns that require it.
        :return: Dataframe containing all the scraped reviews
        """
        # Setting up Chrome driver
        options = webdriver.ChromeOptions()
        options.add_argument("--window-size=800,1200")
        driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH)

        hotel_reviews_scraped = []

        cities_urls = []
        with open('assets/urls/agodaUrls.txt') as webSet:
            for url in webSet:
                cities_urls.append(url.replace('\n', ''))
        for url in cities_urls:
            driver.get(url)
            time.sleep(10)
            # Scrape basic info before getting the reviews
            hotel_name = driver.find_element_by_xpath(
                '//h1[@data-selenium="hotel-header-name"]').text
            hotel_address = driver.find_element_by_xpath('//span[@data-selenium="hotel-address-map"]').text.replace(
                '\n', '')
            total_reviews = driver.find_element_by_xpath('//*[@id="reviewSection"]/div[2]/span[1]').text.split()[
                2].strip('()')
            hotel_lat = float(driver.find_element_by_xpath('/html/head/meta[7]').get_attribute('content'))
            hotel_lng = float(driver.find_element_by_xpath('/html/head/meta[8]').get_attribute('content'))
            average_score = float(
                driver.find_element_by_xpath('//span[@data-selenium="hotel-header-review-score"]').text)
            reviews = driver.find_elements_by_class_name('Review-comment')

            # Loop over all review elements and gather information
            for review in reviews:
                review_score = float(review.find_element_by_class_name('Review-comment-leftScore').text)
                try:
                    review_nationality = review.find_element_by_class_name(
                        'Review-comment-reviewer').find_elements_by_tag_name('span')[-1].text
                except:
                    review_nationality = 'Unknown'
                reviews_mixed = review.find_element_by_class_name('Review-comment-bodyText')
                if review_score < 5.5:
                    review_negative = reviews_mixed.text.replace('\n', '').encode('ascii', 'ignore').decode('ascii')
                    review_positive = 'No Positive'
                else:
                    review_positive = reviews_mixed.text.replace('\n', '').encode('ascii', 'ignore').decode('ascii')
                    review_negative = 'No Negative'
                review_total_positive = len(review_positive.split())
                review_total_negative = len(review_negative.split())
                review_date = review.find_element_by_class_name('Review-statusBar-date').text.strip('Reviewed ')
                review_date = pd.to_datetime(review_date, format='%B %d, %Y')
                days_since_review = (TODAY - review_date).days

                hotel_reviews_scraped.append(
                    [hotel_address, 10, review_date, average_score, hotel_name, review_nationality, review_negative,
                     review_total_negative, total_reviews, review_positive, review_total_positive,
                     51, review_score, 'no tags', days_since_review, hotel_lat, hotel_lng])

        # Convert the scraped reviews list to a DataFrame
        hotel_reviews_scraped = pd.DataFrame(hotel_reviews_scraped,
                                             columns=['Hotel_Address', 'Additional_Number_of_Scoring', 'Review_Date',
                                                      'Average_Score', 'Hotel_Name', 'Reviewer_Nationality',
                                                      'Negative_Review', 'Review_Total_Negative_Word_Counts',
                                                      'Total_Number_of_Reviews', 'Positive_Review',
                                                      'Review_Total_Positive_Word_Counts',
                                                      'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score',
                                                      'Tags', 'days_since_review',
                                                      'lat', 'lng'])
        driver.quit()
        return hotel_reviews_scraped
