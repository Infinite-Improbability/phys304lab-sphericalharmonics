import csv
from dataclasses import dataclass

conversion_factor = 9500 # nanoteslas per volt

# Import csv and turn into a list of lists, where each inner list is a row.
raw = []
with open("ugly_data.csv") as f:
     reader = csv.reader(f)
     for row in reader:
         raw.append(row)

# Split into chunks for latitude groupings 
chunks = []
for i in range(1, 27, 4):
    chunk = raw[i:i+3]
    for r in range(0, len(chunk)):
        chunk[r] = chunk[r][0:19]
    chunks.append(chunk)


@dataclass
class PointData:
    latitude: int
    longitude: int
    x: list[float]
    z: list[float]

    def export(self):
        xRange = abs(self.x[2]-self.x[1])
        zRange = abs(self.z[2]-self.z[1])
        return [self.latitude, self.longitude, self.x[0], self.z[0], xRange, zRange]


data = []
for c in chunks:
    lat = int(c[0][0])
    longs= c[0][1::3]
    x = c[1][1:]
    z = c[2][1:]


    for i in range(0, len(longs)):
        try:
            longitude = int(longs[i])
        except ValueError:
            continue

        # grab [avg, min, max] for point and convert from potential (volts) to mag. field (nanoteslas)
        xPoint = [float(el)*conversion_factor for el in x[i*3:i*3+3]]
        zPoint = [float(el)*conversion_factor for el in z[i*3:i*3+3]]

        data.append(PointData(lat, longitude, xPoint, zPoint))

print(data)

with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for point in data:
        writer.writerow(point.export())