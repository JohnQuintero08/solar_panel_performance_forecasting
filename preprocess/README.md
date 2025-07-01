# PREPROCESS OVERVIEW

- The data extracted from the API is transformed into a dataframe and the columns are renamed.
- The date is formated as date type.
- The dataframe is saved in feather format for further processes.
- The rest of the data extracted has doesn't have any wrong type.
- In the insights section was found some data that is out of boudaries. These irregular data is cleaned in this section. Negative values for irrandiance, wind speed and precipitation are not allowed.
