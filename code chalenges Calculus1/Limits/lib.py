def lim(target,fx,spacing,dir):
  
  x_y_left=[ [(target-10),fx(target-10)]]

  
  x_y_right = [  [(target+10),fx(target+10)]  ]
 
  loop_index = int(10 / spacing)

  for i in range(loop_index):
     x_value_left = x_y_left[-1][0] + spacing;
     x_y_left.append([x_value_left,fx(x_value_left)])
     
     x_value_right = x_y_right[-1][0] - spacing;
     x_y_right.append([x_value_right,fx(x_value_right)])
  
  if(dir=='+'):
     return x_y_right;

  elif(dir=='-'):
     return x_y_left
  else:
     if(dir=='+-'):
        return {
           "left":x_y_left,
           "Right":x_y_right
        }
     else:
        print("inacceptable direction")
        return None;