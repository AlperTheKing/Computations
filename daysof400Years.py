import datetime
import pandas as pd

# Initialize a dictionary to hold counts for each day of the month and each weekday
counts = {
    day: {weekday: 0 for weekday in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
    for day in range(1, 32)
}

# Define the 400-year range
start_year = 2000
end_year = start_year + 400  # 2000 to 2399 inclusive

# Function to determine if a year is a leap year in the Gregorian calendar
def is_leap_year(year):
    return (year % 400 == 0) or (year % 4 == 0 and year % 100 != 0)

# Iterate through each year, month, and day to count weekday occurrences
for year in range(start_year, end_year):
    for month in range(1, 13):
        # Determine the number of days in the current month
        if month == 2:
            days_in_month = 29 if is_leap_year(year) else 28
        elif month in [4, 6, 9, 11]:
            days_in_month = 30
        else:
            days_in_month = 31
        
        for day in range(1, days_in_month + 1):
            # Create a date object for the current day
            current_date = datetime.date(year, month, day)
            # Get the name of the weekday
            weekday = current_date.strftime('%A')
            # Increment the count for this day and weekday
            counts[day][weekday] += 1

# Convert the counts dictionary into a Pandas DataFrame for better visualization
df = pd.DataFrame(counts).T  # Transpose to have days as rows
df = df[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]  # Order the weekdays

# Display the DataFrame
print("Frequency of Each Weekday for Each Day of the Month over 400 Years:")
print(df)