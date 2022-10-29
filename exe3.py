from itertools import permutations
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
from datetime import date
import itertools
import math
from typing import List, Any
import yfinance as yf
import matplotlib.pyplot as plt


class PortfolioBuilder:

    def __init__(self):
        self.the_x_vectors = []
        self.res_list = []
        self.b_list = []
        self.b0_vector = []
        self.s_w = []
        self.tickers_list = None
        self.portfolio_quantization = None

    def get_daily_data(self, tickers_list: List[str], start_date, end_date=date.today()) -> pd.DataFrame:
        try:
            start_date = start_date
            end_date = end_date
            df = web.DataReader(tickers_list, 'yahoo', start_date, end_date)
            self.tickers_list = tickers_list
            self.num_of_days = len(df)
            self.data = df["Adj Close"]
            self.data_matrix = self.data.to_numpy()
            self.num_of_brands = len(tickers_list)
            self.x_list = self.generate_x_vectors()
            if self.data.isnull().values.any():
                raise ValueError
            return self.data
        except Exception:
            raise ValueError
        except KeyError :
            raise ValueError
        except ValueError :
            raise ValueError
        except Exception :
            raise ValueError
        except ConnectionError :
            raise ValueError
        except urllib3.exceptions :
            raise ValueError

    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:

        self.portfolio_quantization = portfolio_quantization
        self.generate_x_vectors()
        self.generate_bw_vectors(portfolio_quantization)
        self.res_list = self.rec_permutation(self.num_of_brands, portfolio_quantization)
        x_vec = np.asarray(self.the_x_vectors)
        # self.calculate_S_T()
        days = list(range(2,self.num_of_days))
        aggregator = []
        X = self.generate_x_vectors ()
        X = [ x for x in X if type (x) != type (None) ]
        b0_vector = self.num_of_brands * [1]
        equally_dist_vec = [i * (1 / self.num_of_brands) for i in b0_vector]
        first_wealth = self.calculate_S_T(equally_dist_vec, X[:1])
        aggregator.append(1)
        aggregator.append(first_wealth)
        # b_aggregator.append(self.calculate_S_T())
        for day in days:
            b_t_plus_one = self.bt_for_next_day_for_universal(day)
            wealth = self.calculate_S_T(b_t_plus_one, X[:day])
            aggregator.append(wealth)

        return aggregator



    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:
        b_t = np.asarray ([ 1 / self.num_of_brands for i in range (self.num_of_brands) ])
        w_t = 1
        b_list = [ b_t ]
        wealth_list = [ w_t ]
        for td in range (1, self.num_of_days) :
            x_t = self.x_list [ td ]
            w_t = w_t * b_t.dot (x_t)
            b_t = (b_t * np.exp ((learn_rate * x_t) / (b_t.dot (x_t)))) / (
                np.sum ([ b_t [ k ] * np.exp ((learn_rate * x_t [ k ]) / (b_t.dot (x_t))) for k in
                          range (self.num_of_brands) ]))
            b_list.append (b_t)
            wealth_list.append (w_t)
        return wealth_list

    # this section belongs to find universal portfio
    def rec_permutation(self, tickers_length, const):

        creation = np.arange(0, 1 + (1 / const), 1 / const)
        if tickers_length == 1:
            return [[j] for j in creation]
        res_list = []
        b_list = []
        smaller_perm = self.rec_permutation(tickers_length - 1, const)
        for elem in creation:
            for i in smaller_perm:
                res_list.append([elem] + i)
        return res_list

    def generate_bw_vectors(self, portfolio_quantization):

        tmp_list = []
        len_of_brands = self.num_of_brands
        res_list = self.rec_permutation(len_of_brands, portfolio_quantization)
        self.b0_vector = len_of_brands * [1]

        equally_dist_vec = [i * (1 / len_of_brands) for i in self.b0_vector]

        for perm in res_list:
            isEquallyDist = True
            for i, x in enumerate(perm):
                if x != equally_dist_vec[i]:
                    isEquallyDist = False
            if isEquallyDist:
                self.b_list.append(equally_dist_vec)
                res_list.remove(perm)
        for i in range(len(res_list)):
            if sum(res_list[i]) >= 0.9999999 and sum(res_list[i]) <= 1.0000001:
                self.b_list.append((res_list[i]))
        return self.b_list

    def generate_x_vectors(self):
        x_list = [None]
        for i in range(self.num_of_days - 1):
            x = self.data_matrix[i + 1, :] / self.data_matrix[i, :]
            x_list.append(x)
        return x_list

    def calculate_S_T(self,portfolio_b,X):
        """
        X is a list of xt vecotrs
        :param portfolio_b: is a vector b^w
        :return:"""
        # X = self.generate_x_vectors()
        s0 = 1
        # b=np.asarray([1 / self.num_of_brands for i in range(self.num_of_brands)])
        b_omega = np.asarray(portfolio_b)
        # X_as_np = [np.asarray(x) for x in X]
        acc = 1
        for x in X:
            acc = acc * (x.dot(b_omega))
        return s0 * acc

    def bt_for_next_day_for_universal(self,day):
        summary_b = np.asarray([0] * self.num_of_brands)
        sum_S_ts = 0
        X = self.generate_x_vectors()
        X = [x for x in X if type(x)!=type(None)]
        for portfolio_b in self.b_list:
            S_t = self.calculate_S_T(portfolio_b, (X[:day]))
            summary_b = np.add(summary_b,np.multiply(S_t, portfolio_b))  # mone
            sum_S_ts = sum_S_ts + S_t # mehane

        b_day_plus_one = np.multiply(summary_b, 1 / sum_S_ts)
        return b_day_plus_one

    def return_value(self):
        return pd.DataFrame(self)


if __name__ == '__main__':
    pb = PortfolioBuilder()
    portfolio_quantization = 10
    first = pb.get_daily_data(['GOOG', 'AAPL'], date(2020, 1, 1), date(2020, 1, 6))
    print(first)

    universal = pb.find_universal_portfolio(portfolio_quantization)
    print(universal)



   # print(pb.find_exponential_gradient_portfolio())

    # first1 = plt.plot (first)
    # plt.show (first1)
