from pydantic import BaseModel


class Input(BaseModel):
	#Pydantic model of the ML input
		age: float
		sex: float
		bmi: float
		bp: float
		s1: float
		s2: float
		s3: float
		s4: float
		s5: float
		s6: float
	
