
class A(object):
    def __init__(self,s1,s2='s2'):
        print(s1+s2)
        pass


a_i = A(s2 = 's22')
a = a_i('s1')
