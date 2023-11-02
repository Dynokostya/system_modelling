import pandas as pd
from scipy import stats


def load_data():
    # https://www.itl.nist.gov/div898/education/anova/newcar.dat#:~:text=URL%3A%20https%3A%2F%2Fwww
    # Interest rates of new cars in different cities
    data = {
        "Interest Rate": [
            13.75, 13.50, 13.00, 13.00, 12.75, 12.50, 14.25, 13.00, 12.75,
            12.50, 12.50, 12.40, 12.30, 11.90, 11.90, 14.00, 14.00, 13.51,
            13.50, 13.50, 13.25, 13.00, 12.50, 12.50, 15.00, 14.00, 13.75,
            13.59, 13.25, 12.97, 12.50, 12.25, 11.89, 14.50, 14.00, 14.00,
            13.90, 13.75, 13.25, 13.00, 12.50, 12.45, 13.50, 12.25, 12.25,
            12.00, 12.00, 12.00, 12.00, 11.90, 11.90
        ],
        "City": [
            1, 1, 1, 1, 1, 1, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 6, 6, 6,
            6, 6, 6, 6, 6, 6
        ]
    }

    # Create the DataFrame
    df_new_car = pd.DataFrame(data)
    return df_new_car


def calculate_anova(df_new_car: pd.DataFrame, alpha=0.01):
    # Calculate the overall mean (mean of means)
    overall_mean = df_new_car['Interest Rate'].mean()

    # Conduct ANOVA using the first five cities
    anova_data = [df_new_car['Interest Rate'][df_new_car['City'] == i] for i in range(1, 6)]

    # Calculate the mean of each group
    group_means = [group.mean() for group in anova_data]

    # Calculate the number of observations in each group
    n_groups = [len(group) for group in anova_data]
    total_n = sum(n_groups)

    # Calculate the Between-Group Sum of Squares (SSB)
    ssb = sum(n * (mean - overall_mean) ** 2 for n, mean in zip(n_groups, group_means))

    # Calculate the Between-Group Mean Square (MSB)
    df_between = len(anova_data) - 1
    msb = ssb / df_between

    # Calculate the Within-Group Sum of Squares (SSW)
    ssw = sum(sum((x - mean) ** 2 for x in group) for group, mean in zip(anova_data, group_means))

    # Calculate the Within-Group Mean Square (MSW)
    df_within = total_n - len(anova_data)
    msw = ssw / df_within

    # Calculate the F statistic
    f_statistic = msb / msw

    # Find the critical F value at the 0.01 significance level
    f_critical = stats.f.ppf(1 - alpha, df_between, df_within)
    return overall_mean, msb, f_statistic, f_critical


def main():
    data = load_data()
    overall_mean, msb, f_statistic, f_critical = calculate_anova(data)

    print("Overall mean:", overall_mean)
    print("Between-Group Mean Square:", msb)
    print("F statistic:", f_statistic)
    print("F critical:", f_critical)
    print("Effect is critical") if f_statistic > f_critical else print("Effect is not critical")


main()
