# main
if __name__ == '__main__':

    nt  = input('number of trials  > ')

    ns = 0

    for i in range(1,nt+1):

        print 'trial ' + str(i) + ':'
        print 'negative(a) or positive(s) > '

        trial = raw_input()

        if trial == 's':
            ns += 1.0
        elif trial == 'a':
            ns += 0.0
        else:
            print 'input error!'
            break

    print ns

    sl = ns/nt * 100
    print 'success rate: ' + str(sl) + '[%]'
