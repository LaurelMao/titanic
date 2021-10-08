clear all
train= readtable('C:\Users\33192\Desktop\train.csv');
test = readtable('C:\Users\33192\Desktop\test.csv');

pclass = train.Pclass;
sex = train.Sex;
age = train.Age;
%--------sexuality---------
f_survive=0; m_survive=0;
for i=1:length(sex)
    if strcmp(sex{i}, 'female')
        sex{i} = 0 ; 
        if train.Survived(i)==1
            f_survive=f_survive+1;
        end
    else sex{i} = 1;
        if train.Survived(i)==1
            m_survive=m_survive+1;
        end  
    end
end
sex = cell2mat(sex);
num_f = sum(sex(:)==0); 
num_m = sum(sex(:)==1);
per_sex = [num_f/891 , num_m/891];
pro_sex = [f_survive./num_f, m_survive./num_m ];
pro_survive = sum(train.Survived)./891;
%---------age-----------
mean_age = nanmean(age);
for i=1:length(age)
    if isnan(age(i))
        age(i) = mean_age;
    end
end
age_max=max(age);
age_min=min(age);
gap_age=(age_max-age_min)/10;
age_sort=ones(11,1);
age_sort_survive=ones(11,1);
sec_age=zeros(11,1);
for j=1:11
    sec_age(j)=age_min+(j-1)*gap_age;
for i=1:length(age)
    if age(i)>=sec_age(j)&&age(i)<sec_age(j)+gap_age
        age_sort(j)=age_sort(j)+1;
        if train.Survived(i)==1
            age_sort_survive(j)=age_sort_survive(j)+1;
        end
    end   
end
end
per_age = age_sort./891;
pro_age = age_sort_survive./age_sort;
% bar(age_min:gap_age:age_max,per_age,'y'); 
%------Pclass-------
num_first = sum(pclass(:)==1); 
num_second = sum(pclass(:)==2);
num_third = sum(pclass(:)==3); 
num_class=[sum(pclass(:)==1),sum(pclass(:)==2),sum(pclass(:)==3)];
First_survive=0; Second_survive=0; Third_survive=0;
for i=1:length(pclass)
    if pclass(i) == 1 && train.Survived(i)==1
        First_survive = First_survive + 1;
    end
    if pclass(i) == 2 && train.Survived(i)==1
            Second_survive =  Second_survive +  1;
    end
    if pclass(i) == 3 && train.Survived(i)==1
        Third_survive = Third_survive + 1;
    end
end
per_class = num_class./891;
pro_class = [First_survive./num_first, Second_survive./num_second,Third_survive./num_third];
%-----predict-------
predict = zeros(length(test.Sex),1);
for i=1:length(test.Sex)
    if strcmp(test.Sex{i}, 'female')
        test.Sex{i} = 0 ; 
    else test.Sex{i} = 1;
    end
end
sex_t = cell2mat(test.Sex);
for i=1:length(test.Age) 
    %age·Ö¶Î
 for k=1:10
      if test.Age(i)>=sec_age(k)&&test.Age(i)<sec_age(k+1)
          k_age=k;
      end
 end
    if test.Age(i) == age_max
        k_age=11;
    end
P1 = exp(log(pro_survive) + log(pro_age(k_age)) + log(pro_class(test.Pclass(i))) + log(pro_sex(sex_t(i)+1)));  
P2 = exp(log(per_sex(sex_t(i)+1)) + log(per_class(test.Pclass(i))) + log(per_age(k_age)));
P = P1./P2;
P1n = exp(log(1-pro_survive) + log(1-pro_age(k_age)) + log(1-pro_class(test.Pclass(i))) + log(1-pro_sex(sex_t(i)+1)));
P2n = exp(log(1-per_sex(sex_t(i)+1)) + log(1-per_class(test.Pclass(i))) + log(1-per_age(k_age)));
Pn = P1n./P2n;
if P > Pn
    predict(i) = 1;
else predict(i) = 0;
end
end
passenger_id=(1:length(test.Sex))';
title = {'PassengerId','Survived'};
result_table=table(passenger_id,predict,'VariableNames',title);
writetable(result_table, 'Submission.csv');
