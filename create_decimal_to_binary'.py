# Function to print binary number for the 
# input decimal using recursion 
def decimalToBinary(n): 

	if(n > 1): 
		# divide with integral result 
		# (discard remainder) 
		decimalToBinary(n//2) 

	
	print(str(n%2) +",", end=' ') 
	
	

# Driver code 
if __name__ == '__main__': 
    for i in range(0,32):
	    decimalToBinary(i) 
	    print("\n") 


